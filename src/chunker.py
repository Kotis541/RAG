import json.encoder
from typing import Dict, Any, Generator
from .parser import RagParser
import ast
import re


def yield_buffer(file_path: str, buffer: list, size: int) -> Generator[Dict[str, Any], None, None]:
    """Helper to yield chunks from a buffer of text segments."""
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


def _line_offsets(content):
    offset = [0]
    for i, ch in enumerate(content):
        if ch == '\n':
            offset.append(i + 1)
    return offset

def _node_range(offsets, node):
    start = offsets[node.lineno - 1] + node.col_offset
    end = offsets[node.end_lineno - 1] + node.end_col_offset
    return start, end

class RagChunker:
    @staticmethod
    def chunk_files(file: Dict[str, str], size: int) -> Generator[Dict[str, Any], None, None]:
        start_index = 0
        content = file["content"]
        content_len = len(content)

        if ".md" in file['file_path']:
            segments = []
            pos = 0
            while pos < content_len:
                try:
                    boundary = content.index("\n\n", pos)
                    seq_end = boundary + 2
                except ValueError:
                    seq_end = content_len

                if seq_end - pos > size:
                    sub = pos
                    while sub < seq_end:
                        end = min(sub + size, seq_end)
                        if end < seq_end:
                            nl = content.rfind('\n', sub, end)
                            if nl > sub:
                                end = nl + 1
                        segments.append({'text': content[sub:end], 'start': sub})
                        sub = end
                else:
                    segments.append({'text': content[pos:seq_end], 'start': pos})
                pos = seq_end

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
    def _flatten_nodes(nodes, offsets, size):
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
        chunk_text = content[start:end]
        # Jména přímo v chunku
        names = re.findall(r'(?:class|def)\s+(\w+)', chunk_text)
        
        # Najdi enclosing class hledáním zpětně od začátku chunku
        prefix = content[:start]
        classes = re.findall(r'^class\s+(\w+)', prefix, re.MULTILINE)
        if classes:
            names.append(classes[-1])  # poslední class před chunkem = enclosing
        
        return ' '.join(names)

                
