import bm25s
from typing import Any
import pickle
import os
import re
from typing import List, Dict
from functools import lru_cache
import chromadb
from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Load and cache the sentence-transformer embedding model (singleton)."""
    return SentenceTransformer('all-MiniLM-L6-v2')


def _split_identifiers(text: str) -> str:
    """Expand camelCase and snake_case identifiers into space-separated words for better BM25 tokenization."""
    def expand_camel(match):
        word = match.group(0)
        parts = re.sub(r'([a-z])([A-Z])', r'\1 \2', word)
        parts = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', parts)
        return word + ' ' + parts

    result = re.sub(r'[a-zA-Z]+_[a-zA-Z_]+', lambda m: m.group(0) + ' ' + m.group(0).replace('_', ' '), text)
    result = re.sub(r'[a-z]+[A-Z][a-zA-Z]+|[A-Z][a-z]+(?:[A-Z][a-z]+)+|[A-Z]{2,}[a-z][a-zA-Z]*', expand_camel, result)
    return result.lower()


def tokenize(chunks: list[dict[str, Any]], save_path: str = "data/processed"):
    """Build and persist the BM25 index and Chroma vector store from a list of chunk dicts."""
    model = _get_model()
    corpus = [
        _split_identifiers(
            chunk['file_path'] + '\n'
            + chunk['file_path'] + '\n'
            + chunk['file_path'] + '\n'
            + chunk.get('class_name', ' ') + '\n'
            + chunk['content']
        )
        for chunk in chunks
    ]

    db_path = os.path.join("data", "processed", "chroma_db")
    chroma_client = chromadb.PersistentClient(db_path)
    collection = chroma_client.get_or_create_collection(name="vllm_docs_code")

    ids = [
        f"{c['file_path']}: {c['first_character_index']}:{c['last_character_index']}"
        for c in chunks
    ]
    embeddings = model.encode(corpus, show_progress_bar=True).tolist()

    batch_size = 5000
    for i in range(0, len(corpus), batch_size):
        collection.add(
            documents=corpus[i:i + batch_size],
            ids=ids[i:i + batch_size],
            embeddings=embeddings[i:i + batch_size]
        )

    corpus_tokens = bm25s.tokenize(corpus)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)

    os.makedirs(save_path, exist_ok=True)
    retriever.save(save_path)
    with open(os.path.join(save_path, "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)


class BM25Searcher:
    """Loads a persisted BM25 index and provides single and batch keyword search."""

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
        query_tokens = bm25s.tokenize(_split_identifiers(query))
        results, _ = self.retriever.retrieve(query_tokens, k=k)
        return [self.all_chunks[idx] for idx in results[0]]

    def search_batch(self, queries: List[str], k: int) -> List[List[Dict[str, Any]]]:
        """Return top-k chunk dicts for each query in a batch."""
        if self.retriever is None or self.all_chunks is None:
            return []
        processed = [_split_identifiers(q) for q in queries]
        query_tokens = bm25s.tokenize(processed)
        results, _ = self.retriever.retrieve(query_tokens, k=k)
        return [[self.all_chunks[idx] for idx in results[i]] for i in range(len(queries))]


class ChromaSearcher:
    """Loads a persisted Chroma vector store and provides single and batch semantic search."""

    def __init__(self):
        """Connect to the Chroma database and load the embedding model."""
        self.path = os.path.join("data", "processed", "chroma_db")
        self.client = chromadb.PersistentClient(self.path)
        self.collection = self.client.get_collection("vllm_docs_code")
        self.transformer = _get_model()

    @lru_cache(maxsize=512)
    def search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Return the top-k chunk dicts most semantically similar to query."""
        embedding = self.transformer.encode([query]).tolist()
        results = self.collection.query(query_embeddings=embedding, n_results=k)
        chunks = []
        for doc_id in results['ids'][0]:
            file_path, start, end = doc_id.rsplit(':', 2)
            chunks.append({
                'file_path': file_path,
                'first_character_index': int(start),
                'last_character_index': int(end),
            })
        return chunks

    def search_batch(self, queries: List[str], k: int) -> List[List[Dict[str, Any]]]:
        """Return top-k semantically similar chunk dicts for each query in a batch."""
        embeddings = self.transformer.encode(queries).tolist()
        results = self.collection.query(query_embeddings=embeddings, n_results=k)
        all_results = []
        for ids_per_query in results['ids']:
            chunks = []
            for doc_id in ids_per_query:
                file_path, start, end = doc_id.rsplit(':', 2)
                chunks.append({
                    'file_path': file_path,
                    'first_character_index': int(start),
                    'last_character_index': int(end),
                })
            all_results.append(chunks)
        return all_results


def _reciprocal_rank_fusion(bm25_results: List[Dict], chroma_results: List[Dict], k: int) -> List[Dict]:
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