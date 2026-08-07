from pathlib import Path
from typing import List, Dict
import os


class RagParser:
    """Walks a repository directory and loads all .py and .md files."""

    @staticmethod
    def discover_files(repo_path: str) -> List[Dict[str, str]]:
        """Return a list of {file_path, content} dicts for every .py/.md file found (excluding tests/)."""
        base_dir = Path(repo_path)
        documents = []

        if not base_dir.exists() or not base_dir.is_dir():
            raise FileNotFoundError(f"file '{repo_path}' doesn't exist!")

        for root, dirs, files in os.walk(str(base_dir), followlinks=True):
            parts = Path(root).parts
            if "tests" in parts:
                dirs.clear()
                continue
            for filename in files:
                path = Path(root) / filename
                if path.suffix not in (".py", ".md"):
                    continue
                try:
                    content = path.read_text(encoding="utf-8")
                    documents.append({
                        "file_path": str(path),
                        "content": content
                    })
                except Exception as e:
                    print(f"Can't read {path} cause of {e}")

        return documents
