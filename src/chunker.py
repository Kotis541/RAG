from typing import Dict, Any, Generator
import ast
import re


def yield_buffer(file_path: str, buffer: list, size: int) -> Generator[Dict[str, Any], None, None]:
    """Yield fixed-size chunks from a list of text segments, respecting segment boundaries."""
    current_chunk_segments = []
    current_chunk_len = 0

    for segment in buffer:
        if current_chunk_len + len(segment['text']) > size and current_chunk_segments:
            yield {
                "file_path": file_path,
                "first_character_index": current_chunk_segments[0]['start'],
                "last_character_index": current_chunk_segments[-1]['start'] + len(current_chunk_segments[-1]['text']),
            }
            current_chunk_segments, current_chunk_len = [], 0
        current_chunk_segments.append(segment)
        current_chunk_len += len(segment['text'])

    if current_chunk_segments:
        yield {
            "file_path": file_path,
            "first_character_index": current_chunk_segments[0]['start'],
            "last_character_index": current_chunk_segments[-1]['start'] + len(current_chunk_segments[-1]['text']),
        }


def _line_offsets(content: str) -> list:
    """Return a list of character offsets for the start of each line in content."""
    offset = [0]
    for i, ch in enumerate(content):
        if ch == '\n':
            offset.append(i + 1)
    return offset


def _node_range(offsets: list, node: ast.AST) -> tuple:
    """Return the (start, end) character range of an AST node using precomputed line offsets."""
    start = offsets[node.lineno - 1] + node.col_offset
    end = offsets[node.end_lineno - 1] + node.end_col_offset
    return start, end


class RagChunker:
    """Splits source files into overlapping text chunks suitable for indexing."""

    @staticmethod
    def chunk_files(file: Dict[str, str], size: int) -> Generator[Dict[str, Any], None, None]:
        """Yield character-range chunks for a single file, using AST boundaries for .py and headings for .md."""
        content = file["content"]
        content_len = len(content)

        if ".md" in file['file_path']:
            segments = []

            heading_positions = [0]
            for match in re.finditer(r'(?:^|\n)(#{1,6} )', content):
                pos = match.start(1) if match.start(0) == 0 else match.start(0) + 1
                if pos > 0:
                    heading_positions.append(pos)
            heading_positions.append(content_len)

            for i in range(len(heading_positions) - 1):
                sec_start = heading_positions[i]
                sec_end = heading_positions[i + 1]
                section_text = content[sec_start:sec_end]
                section_len = sec_end - sec_start

                if section_len > size:
                    # Section too large: split at newline boundaries
                    sub = sec_start
                    while sub < sec_end:
                        end = min(sub + size, sec_end)
                        if end < sec_end:
                            nl = content.rfind('\n', sub, end)
                            if nl > sub:
                                end = nl + 1
                        segments.append({'text': content[sub:end], 'start': sub})
                        sub = end
                else:
                    segments.append({'text': section_text, 'start': sec_start})

            yield from yield_buffer(file['file_path'], segments, size)

        elif ".py" in file['file_path']:
            try:
                nodes = ast.parse(content).body
                buffer = []
                buffer_len = 0
                offsets = _line_offsets(content)
                flat_ranges = RagChunker._flatten_nodes(nodes, offsets, size)

                cursor = 0

                for node_start, node_end in flat_ranges:
                    if node_start > cursor:
                        inter_text = content[cursor:node_start]
                        buffer.append({'text': inter_text, 'start': cursor})
                        buffer_len += len(inter_text)

                    segment_text = content[node_start:node_end]
                    segment_len = len(segment_text)
                    if segment_len > size:
                        if buffer:
                            yield from yield_buffer(file['file_path'], buffer, size)
                            buffer, buffer_len = [], 0

                        line_start = 0
                        while line_start < segment_len:
                            search_start = line_start + size
                            line_end = segment_text.rfind('\n', line_start, search_start)
                            if line_end == -1 or line_end <= line_start:
                                line_end = min(line_start + size, segment_len)

                            yield {
                                "file_path": file["file_path"],
                                "first_character_index": node_start + line_start,
                                "last_character_index": node_start + line_end,
                            }
                            line_start = line_end + 1
                    else:
                        if buffer_len + segment_len > size:
                            yield from yield_buffer(file['file_path'], buffer, size)
                            buffer, buffer_len = [], 0
                        buffer.append({'text': segment_text, 'start': node_start})
                        buffer_len += segment_len
                    cursor = node_end

                if buffer:
                    yield from yield_buffer(file['file_path'], buffer, size)
            except SyntaxError:
                if content:
                    yield {
                        "file_path": file["file_path"],
                        "first_character_index": 0,
                        "last_character_index": len(content)
                    }

    @staticmethod
    def _flatten_nodes(nodes: list, offsets: list, size: int) -> list:
        """Recursively flatten AST nodes into (start, end) ranges, splitting large nodes into their children."""
        result = []
        for node in nodes:
            if not hasattr(node, 'end_lineno') or node.end_lineno is None:
                continue
            start, end = _node_range(offsets, node)
            node_size = end - start

            if node_size > size and hasattr(node, 'body') and node.body:
                body_ranges = RagChunker._flatten_nodes(node.body, offsets, size)
                if body_ranges:
                    if body_ranges[0][0] > start:
                        result.append((start, body_ranges[0][0]))
                    result.extend(body_ranges)
                    if body_ranges[-1][1] < end:
                        result.append((body_ranges[-1][1], end))
                    continue

            result.append((start, end))
        return result

    @staticmethod
    def _extract_names(content: str, start: int, end: int) -> str:
        """Extract class/def names defined in a chunk plus the enclosing class name (if any)."""
        chunk_text = content[start:end]
        names = re.findall(r'(?:class|def)\s+(\w+)', chunk_text)

        prefix = content[:start]
        classes = re.findall(r'^class\s+(\w+)', prefix, re.MULTILINE)
        if classes:
            names.append(classes[-1])

        return ' '.join(names)
