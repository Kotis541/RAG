from .chunker import RagChunker
from .parser import RagParser
from .index import tokenize, BM25Searcher
from .llm import LLMGenerator
from .models import StudentSearchResults, MinimalSearchResults, RagDataset, StudentSearchResultsAndAnswer, MinimalAnswer, MinimalSource
import os
import json

class RagPipeline:
    """CLI for the Retrieval-Augmented Generation system."""

    def __init__(self):
        """Initializes the pipeline by loading the search index and LLM."""
        self.searcher = BM25Searcher()
        self.llm = LLMGenerator()

    def index(self, max_chunk_size: int = 1000):
        """Index the repository."""
        try:
            chunker = RagChunker()
            parser = RagParser()
            docs = parser.load_vocabulary("vllm-0.10.1")
            all_chunks = []

            os.makedirs("data/processed", exist_ok=True)

            for file in docs:
                file_content = file['content']
                for chunk in chunker.chunk_files(file, max_chunk_size):
                    start = chunk["first_character_index"]
                    end = chunk["last_character_index"]

                    chunk_with_content = chunk.copy()
                    chunk_with_content['content'] = file_content[start:end]
                    chunk_with_content['class_name'] = RagChunker._extract_names(file_content, start, end)
                    all_chunks.append(chunk_with_content)
            
            tokenize(all_chunks)
            self.searcher = BM25Searcher()
            return "Ingestion complete! Indices saved under data/processed/"
        except Exception as e:
            print(f"[INDEX ERROR]: {e}")

    
    def search(self, query: str, k: int = 3) -> str:
        top_result = self.searcher.search(query, k)
        return json.dumps(top_result, indent=2)

    
    def search_dataset(self, dataset_path: str, k: int = 5, save_directory: str = "data/output/search_results") -> str:
        os.makedirs(os.path.dirname(save_directory), exist_ok=True)

        with open(dataset_path, 'r') as json_file:
            raw_data = json.load(json_file)

        dataset = RagDataset.model_validate(raw_data)
        questions = [q.question for q in dataset.rag_questions]
        batch_results = self.searcher.search_batch(questions, k)

        search_results = []
        for question_data, top_results in zip(dataset.rag_questions, batch_results):
            sources = [MinimalSource.model_validate(source) for source in top_results]
            result_entry = MinimalSearchResults(
                question_id=question_data.question_id,
                question=question_data.question,
                retrieved_sources=sources,
            )
            search_results.append(result_entry)

        output = StudentSearchResults(search_results=search_results, k=k)

        with open(save_directory, 'w', encoding='utf-8') as f:
            f.write(output.model_dump_json(indent=2))
        return f"Search results saved to {save_directory}"


    def answer(self, query: str, k: int = 1) -> str:
        top_results = self.searcher.search(query, k)

        if not top_results:
            return "No relevant sources found to answer the question."

        chunks_by_file = {}
        for result in top_results:
            path = result['file_path']
            if path not in chunks_by_file:
                chunks_by_file[path] = []
            chunks_by_file[path].append(result)

        context_parts = []
        for path, chunks in chunks_by_file.items():
            print(f"path {path}")
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                for chunk in chunks:
                    start = chunk['first_character_index']
                    end = chunk['last_character_index']
                    context_parts.append(content[start:end])
            except FileNotFoundError:
                print(f"Warning: File {path} not found. Skipping.")
            except Exception as e:
                print(f"Warning: Error reading file {path}: {e}. Skipping.")

        if not context_parts:
            return "Could not load any context to answer the question."

        full_context = "\n\n".join(context_parts)
        answer = self.llm.generate_answer(full_context, query)
        return answer
    
    def answer_dataset(self, student_search_results_path: str, save_directory: str = "data/output/search_results_and_answer", k: int = 5) -> str:
        os.makedirs(os.path.dirname(save_directory), exist_ok=True)

        with open(student_search_results_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        dataset = RagDataset.model_validate(raw_data)
        questions = [q.question for q in dataset.rag_questions]

        # --- Batch Processing ---
        print(f"1/3: Searching for sources for {len(questions)} questions in a batch...")
        batch_top_results = self.searcher.search_batch(questions, k)

        contexts = []
        for i, top_results in enumerate(batch_top_results):
            context_parts = []

            if top_results:
                chunks_by_file = {}
                for result in top_results:
                    path = result['file_path']
                    if path not in chunks_by_file:
                        chunks_by_file[path] = []
                    chunks_by_file[path].append(result)

                for path, chunks in chunks_by_file.items():
                    try:
                        with open(path, 'r', encoding='utf-8') as f_content:
                            content = f_content.read()
                        for chunk in chunks:
                            start = chunk['first_character_index']
                            end = chunk['last_character_index']
                            context_parts.append(content[start:end])
                    except FileNotFoundError:
                        print(f"Warning: File {path} not found. Skipping context for question {i}.")
                    except Exception as e:
                        print(f"Warning: Error reading file {path}: {e}. Skipping context for question {i}.")
            
            if not context_parts:
                contexts.append("Could not load any context to answer the question.")
            else:
                contexts.append("\n\n".join(context_parts))

        print(f"3/3: Generating answers for {len(questions)} questions...")

        answered_results = []
        for i in range(len(questions)):
            question_data = dataset.rag_questions[i]
            top_results = batch_top_results[i]
            context = contexts[i]

            answer = self.llm.generate_answer(context, questions[i])

            sources = [
                MinimalSource.model_validate({key: value for key, value in source.items() if key != 'content'})
                for source in top_results
            ]

            result_entry = MinimalAnswer(
                question_id=question_data.question_id,
                question=question_data.question,
                retrieved_sources=sources,
                answer=answer,
            )
            answered_results.append(result_entry)

            # Save after every answer
            final_output = StudentSearchResultsAndAnswer(search_results=answered_results, k=k)
            with open(save_directory, 'w', encoding='utf-8') as f:
                f.write(final_output.model_dump_json(indent=2))
            print(f"  [{i+1}/{len(questions)}] Saved to {save_directory}")

        return f"Answered dataset saved to {save_directory}"


    def evaluate(self, student_answer_path: str, dataset_path: str, k: int = 5, max_context_length: int = 2000) -> str:
        """Evaluate search results against ground truth."""
        return "Evaluating search results..."
