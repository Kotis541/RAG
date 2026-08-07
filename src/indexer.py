import bm25s
from typing import Any
import pickle
import os
from typing import List
from functools import lru_cache
import chromadb
from sentence_transformers import SentenceTransformer
from .searchers import _expand_identifiers, _get_embedding_model


def build_index(chunks: list[dict[str, Any]], save_path: str = "data/processed"):
    """Build and persist both the BM25 keyword index and the Chroma vector store from chunk dicts."""
    model = _get_embedding_model()
    corpus = [
        _expand_identifiers(
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
