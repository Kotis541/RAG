from pathlib import Path
from typing import List, Dict

class RagParser:
    @staticmethod
    def load_vocabulary(repo_path: str) -> Dict[int, str]:
        base_dir = Path(repo_path)
        documents = []

        if not base_dir.exists() or not base_dir.is_dir:
            raise FileNotFoundError(f"file '{repo_path}' doesn't exist!")
        
        for path in base_dir.rglob("*"):
            if path.is_file() and path.suffix in [".py", ".md"]:
                try:
                    content = path.read_text(encoding="utf-8")

                    documents.append({
                        "file_path": "data/raw/" + str(path),
                        "content": content
                    })
                except Exception as e:
                    print(f"Can't read {path} cause of {e}")

        return documents