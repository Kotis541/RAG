from transformers.models import qwen2_audio
from torch import chunk
import bm25s
from typing import Any
import pickle
import os
import re
from typing import List, Dict


def _split_identifiers(text: str) -> str:
    """Split camelCase and snake_case identifiers into separate words.
    
    E.g. 'FusedMoEActivationFormat' -> 'FusedMoEActivationFormat Fused Mo E Activation Format'
         'activation_formats' -> 'activation_formats activation formats'
    """
    def expand_camel(match):
        word = match.group(0)
        parts = re.sub(r'([a-z])([A-Z])', r'\1 \2', word)
        parts = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', parts)
        return word + ' ' + parts

    result = re.sub(r'[a-zA-Z]+_[a-zA-Z_]+', lambda m: m.group(0) + ' ' + m.group(0).replace('_', ' '), text)

    result = re.sub(r'[a-z]+[A-Z][a-zA-Z]+|[A-Z][a-z]+(?:[A-Z][a-z]+)+|[A-Z]{2,}[a-z][a-zA-Z]*', expand_camel, result)

    return result.lower()


def tokenize(file: list[dict[str, Any]]):
    corpus = [
        _split_identifiers(
            chunk['file_path'] + '\n' 
            + chunk['file_path'] + '\n'
            + chunk['file_path'] + '\n'
            + chunk.get('class_name', ' ') + '\n'
            + chunk['content']
        )
        for chunk in file
    ]

    corpus_tokens = bm25s.tokenize(corpus)

    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)

    save_path = "data/processed"
    os.makedirs(save_path, exist_ok=True)
    retriever.save(save_path)
    with open(os.path.join(save_path, "chunks.pkl"), "wb") as f:
        pickle.dump(file, f)

class BM25Searcher:
    """A class to handle BM25 index loading and searching."""
    def __init__(self, load_path: str = "data/processed"):
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
        if self.retriever is None or self.all_chunks is None:
            return []

        query_tokens = bm25s.tokenize(_split_identifiers(query))
        results, scores = self.retriever.retrieve(query_tokens, k=k)

        top_k_indices = results[0]
        
        found_results = [self.all_chunks[doc_idx] for doc_idx in top_k_indices]

        return found_results
    

    def search_batch(self, queries: List[str], k: int) -> List[List[Dict[str, Any]]]:
        if self.retriever is None or self.all_chunks is None:
            return []
        
        processed = [_split_identifiers(q) for q in queries]
        query_tokens = bm25s.tokenize(processed)
        results, scores = self.retriever.retrieve(query_tokens, k=k)

        all_results = []
        for i in range(len(queries)):
            found = [self.all_chunks[idx] for idx in results[i]]
            all_results.append(found)
        
        return all_results
        
