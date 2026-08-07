import bm25s
from typing import Any, Protocol, runtime_checkable
import pickle
import os
import re
from typing import List, Dict
from functools import lru_cache
import chromadb
from sentence_transformers import SentenceTransformer


@runtime_checkable
class Searcher(Protocol):
    """Common interface for keyword and semantic search backends."""

    def search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Return the top-k most relevant chunk dicts for a single query."""
        ...

    def search_batch(self, queries: List[str], k: int) -> List[List[Dict[str, Any]]]:
        """Return top-k most relevant chunk dicts for each query in a batch."""
        ...


@lru_cache(maxsize=1)
def _get_embedding_model() -> SentenceTransformer:
    """Load and cache the sentence-transformer embedding model (singleton)."""
    return SentenceTransformer('all-MiniLM-L6-v2')


def _expand_identifiers(text: str) -> str:
    """Expand camelCase and snake_case identifiers into space-separated words for better BM25 recall."""
    def expand_camel(match):
        word = match.group(0)
        parts = re.sub(r'([a-z])([A-Z])', r'\1 \2', word)
        parts = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', parts)
        return word + ' ' + parts

    result = re.sub(r'[a-zA-Z]+_[a-zA-Z_]+', lambda m: m.group(0) + ' ' + m.group(0).replace('_', ' '), text)
    result = re.sub(r'[a-z]+[A-Z][a-zA-Z]+|[A-Z][a-z]+(?:[A-Z][a-z]+)+|[A-Z]{2,}[a-z][a-zA-Z]*', expand_camel, result)
    return result.lower()


class BM25Searcher:
    """Keyword search backend: loads a persisted BM25 index and searches it."""

    def __init__(self, load_path: str = "data/processed"):
        """Load the BM25 retriever and chunk metadata from disk."""
        self.retriever = None
        self.all_chunks = None
        try:
            self.retriever = bm25s.BM25.load(load_path, load_corpus=False)
            with open(os.path.join(load_path, "chunks.pkl"), "rb") as f:
                self.all_chunks = pickle.load(f)
        except FileNotFoundError:
            print("Warning: BM25 index or chunks not found. Please run the 'index' command first.")
        except Exception as e:
            print(f"An error occurred while loading search index: {e}")

    def search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Return the top-k chunk dicts most relevant to query using BM25 scoring."""
        if self.retriever is None or self.all_chunks is None:
            return []
        query_tokens = bm25s.tokenize(_expand_identifiers(query))
        results, _ = self.retriever.retrieve(query_tokens, k=k)
        return [self.all_chunks[idx] for idx in results[0]]

    def search_batch(self, queries: List[str], k: int) -> List[List[Dict[str, Any]]]:
        """Return top-k chunk dicts for each query in a batch."""
        if self.retriever is None or self.all_chunks is None:
            return []
        processed = [_expand_identifiers(q) for q in queries]
        query_tokens = bm25s.tokenize(processed)
        results, _ = self.retriever.retrieve(query_tokens, k=k)
        return [[self.all_chunks[idx] for idx in results[i]] for i in range(len(queries))]


class ChromaSearcher:
    """Semantic search backend: loads a persisted Chroma vector store and searches it."""

    def __init__(self):
        """Connect to the Chroma database and load the embedding model."""
        self.path = os.path.join("data", "processed", "chroma_db")
        self.client = chromadb.PersistentClient(self.path)
        self.collection = self.client.get_collection("vllm_docs_code")
        self.transformer = _get_embedding_model()

    @lru_cache(maxsize=512)
    def search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Return the top-k chunk dicts most semantically similar to query."""
        embedding = self.transformer.encode([query]).tolist()
        results = self.collection.query(query_embeddings=embedding, n_results=k)
        return self._parse_ids(results['ids'][0])

    def search_batch(self, queries: List[str], k: int) -> List[List[Dict[str, Any]]]:
        """Return top-k semantically similar chunk dicts for each query in a batch."""
        embeddings = self.transformer.encode(queries).tolist()
        results = self.collection.query(query_embeddings=embeddings, n_results=k)
        return [self._parse_ids(ids) for ids in results['ids']]

    @staticmethod
    def _parse_ids(ids: List[str]) -> List[Dict[str, Any]]:
        """Convert Chroma document IDs back into chunk dicts."""
        chunks = []
        for doc_id in ids:
            file_path, start, end = doc_id.rsplit(':', 2)
            chunks.append({
                'file_path': file_path,
                'first_character_index': int(start),
                'last_character_index': int(end),
            })
        return chunks


def reciprocal_rank_fusion(bm25_results: List[Dict], chroma_results: List[Dict], k: int) -> List[Dict]:
    """Merge BM25 and Chroma result lists into a single top-k ranking using Reciprocal Rank Fusion."""
    scores = {}
    chunks_map = {}

    for i, chunk in enumerate(bm25_results):
        key = f"{chunk['file_path']}:{chunk['first_character_index']}:{chunk['last_character_index']}"
        scores[key] = scores.get(key, 0) + 1.0 / (i + 60)
        chunks_map[key] = chunk

    for i, chunk in enumerate(chroma_results):
        key = f"{chunk['file_path']}:{chunk['first_character_index']}:{chunk['last_character_index']}"
        scores[key] = scores.get(key, 0) + 0.0 / (i + 60)
        chunks_map[key] = chunk

    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [chunks_map[key] for key in sorted_keys[:k]]
