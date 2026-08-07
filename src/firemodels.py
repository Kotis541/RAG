from .chunker import RagChunker
from .parser import RagParser
from .index import tokenize, BM25Searcher, ChromaSearcher, _reciprocal_rank_fusion
from .llm import LLMGenerator
from .models import StudentSearchResults, MinimalSearchResults, RagDataset, StudentSearchResultsAndAnswer, MinimalSource, MinimalAnswer
import os
import json
from tqdm import tqdm


def _validate_k(k: int, name: str = "k") -> None:
    """Raise ValueError if k is outside the valid range [1, 19]."""
    if k <= 0 or k >= 20:
        raise ValueError(f"{name} must be between 1 and 19, got {k}")


class RagPipeline:
    """CLI for the Retrieval-Augmented Generation system."""

    def __init__(self, index_path: str = "data/processed"):
        """Load the BM25 and Chroma search indexes and initialise the LLM."""
        self.index_path = index_path
        self.searcher = BM25Searcher(load_path=index_path)
        self.llm = LLMGenerator()
        self.chroma_searcher = ChromaSearcher()

    def _load_context(self, top_results: list) -> str:
        """Read source file chunks from disk and join them into a single context string."""
        chunks_by_file: dict = {}
        for result in top_results:
            chunks_by_file.setdefault(result['file_path'], []).append(result)

        context_parts = []
        for path, chunks in chunks_by_file.items():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                for chunk in chunks:
                    context_parts.append(content[chunk['first_character_index']:chunk['last_character_index']])
            except FileNotFoundError:
                print(f"Warning: {path} not found, skipping.")
            except Exception as e:
                print(f"Warning: could not read {path}: {e}, skipping.")

        return "\n\n".join(context_parts) if context_parts else ""

    def _hybrid_search(self, questions: list, k: int) -> list:
        """Run BM25 + Chroma search for each question and fuse results with RRF."""
        bm25_batch = self.searcher.search_batch(questions, k * 2)
        chroma_batch = self.chroma_searcher.search_batch(questions, k * 2)
        return [
            _reciprocal_rank_fusion(bm25_batch[i], chroma_batch[i], k)
            for i in range(len(questions))
        ]


    def index(self, input_path: str = "data/raw/vllm-0.10.1", output_path: str = "data/processed", max_chunk_size: int = 2000):
        """Parse, chunk and index a repository so it can be searched."""
        if max_chunk_size > 2000:
            raise ValueError("max_chunk_size is too big, maximum is 2000!")
        elif max_chunk_size <= 0:
            raise ValueError("max_chunk_size must be more than 0")
        try:
            chunker = RagChunker()
            parser = RagParser()
            docs = parser.load_vocabulary(input_path)
            all_chunks = []

            os.makedirs(output_path, exist_ok=True)

            for file in tqdm(docs, desc="Chunking files", unit="file"):
                file_content = file['content']
                for chunk in chunker.chunk_files(file, max_chunk_size):
                    start = chunk["first_character_index"]
                    end = chunk["last_character_index"]
                    chunk_with_content = chunk.copy()
                    chunk_with_content['content'] = file_content[start:end]
                    chunk_with_content['class_name'] = RagChunker._extract_names(file_content, start, end)
                    all_chunks.append(chunk_with_content)

            tokenize(all_chunks, save_path=output_path)
            self.searcher = BM25Searcher(load_path=output_path)
            return f"Ingestion complete! Indices saved under {output_path}/"
        except Exception as e:
            print(f"[INDEX ERROR]: {e}")

    def search(self, query: str, k: int = 3) -> None:
        """Print the top-k most relevant chunks for a single query."""
        _validate_k(k)
        if not query or not query.strip():
            print("[SEARCH ERROR]: query must not be empty")
            return
        try:
            top_results = self._hybrid_search([query], k)[0]
            for result in top_results:
                print(f"{result['file_path']} [{result['first_character_index']}:{result['last_character_index']}]")
        except Exception as e:
            print(f"[SEARCH ERROR]: {e}")

    def search_dataset(self, dataset_path: str, k: int = 5, save_directory: str = "data/output/search_results.json") -> str:
        """Run hybrid search over every question in a dataset and save the results to JSON."""
        _validate_k(k)
        try:
            parent = os.path.dirname(save_directory)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(dataset_path, 'r') as json_file:
                raw_data = json.load(json_file)
        except FileNotFoundError:
            print(f"[SEARCH ERROR]: dataset file not found: {dataset_path}")
            return
        except json.JSONDecodeError as e:
            print(f"[SEARCH ERROR]: malformed JSON in {dataset_path}: {e}")
            return

        try:
            dataset = RagDataset.model_validate(raw_data)
            questions = [q.question for q in dataset.rag_questions]
            if not questions:
                print("[SEARCH ERROR]: dataset contains no questions")
                return

            batch_results = self._hybrid_search(questions, k)

            search_results = []
            for question_data, top_results in zip(dataset.rag_questions, batch_results):
                sources = [MinimalSource.model_validate(source) for source in top_results]
                search_results.append(MinimalSearchResults(
                    question_id=question_data.question_id,
                    question=question_data.question,
                    retrieved_sources=sources,
                ))

            output = StudentSearchResults(search_results=search_results, k=k)
            with open(save_directory, 'w', encoding='utf-8') as f:
                f.write(output.model_dump_json(indent=2))
            return f"Search results saved to {save_directory}"

        except Exception as e:
            print(f"[SEARCH ERROR]: {e}")

    def answer(self, query: str, k: int = 5) -> str:
        """Answer a single query by retrieving relevant chunks and generating a response."""
        _validate_k(k)
        if not query or not query.strip():
            return "[ANSWER ERROR]: query must not be empty"

        top_results = self.searcher.search(query, k)
        if not top_results:
            return "No relevant sources found to answer the question."

        context = self._load_context(top_results)
        if not context:
            return "Could not load any context to answer the question."

        return self.llm.generate_answer(context, query)

    def answer_dataset(self, search_results_path: str, save_directory: str = "data/output/search_results_and_answer.json", k: int = 5) -> str:
        """Generate answers for all questions in a search-results file and save them incrementally."""
        _validate_k(k)

        try:
            parent = os.path.dirname(save_directory)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(search_results_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
        except FileNotFoundError:
            print(f"[ANSWER ERROR]: file not found: {search_results_path}")
            return
        except json.JSONDecodeError as e:
            print(f"[ANSWER ERROR]: malformed JSON in {search_results_path}: {e}")
            return

        dataset = StudentSearchResults.model_validate(raw_data)
        questions = [q.question for q in dataset.search_results]
        if not questions:
            print("[ANSWER ERROR]: dataset contains no questions")
            return

        print(f"Loaded {len(questions)} questions")
        batch_top_results = self._hybrid_search(questions, k)
        contexts = [self._load_context(top_results) or "Could not load any context to answer the question."
                    for top_results in batch_top_results]

        answered_results = []
        for i in tqdm(range(len(questions)), desc="Generating answers", unit="q"):
            question_data = dataset.search_results[i]
            top_results = batch_top_results[i]

            answer = self.llm.generate_answer(contexts[i], questions[i])
            sources = [
                MinimalSource.model_validate({k: v for k, v in source.items() if k != 'content'})
                for source in top_results
            ]
            answered_results.append(MinimalAnswer(
                question_id=question_data.question_id,
                question=question_data.question,
                retrieved_sources=sources,
                answer=answer,
            ))

            final_output = StudentSearchResultsAndAnswer(search_results=answered_results, k=k)
            with open(save_directory, 'w', encoding='utf-8') as f:
                f.write(final_output.model_dump_json(indent=2))

        print(f"Processed {len(answered_results)} of {len(questions)} questions")
        return f"Answered dataset saved to {save_directory}"

    def evaluate(self, student_answer_path: str, dataset_path: str, k: int = 10, max_context_length: int = 2000) -> str:
        """Evaluate student search results against ground-truth and print Recall@k scores."""
        try:
            with open(student_answer_path, 'r') as f:
                student_results = json.load(f)
        except FileNotFoundError:
            print(f"[EVALUATE ERROR]: file not found: {student_answer_path}")
            return
        except json.JSONDecodeError as e:
            print(f"[EVALUATE ERROR]: malformed JSON in {student_answer_path}: {e}")
            return

        try:
            with open(dataset_path, 'r') as f:
                ground_truth = json.load(f)
        except FileNotFoundError:
            print(f"[EVALUATE ERROR]: file not found: {dataset_path}")
            return
        except json.JSONDecodeError as e:
            print(f"[EVALUATE ERROR]: malformed JSON in {dataset_path}: {e}")
            return

        if StudentSearchResults.model_validate(student_results):
            print("Student answers is valid: True")
        print(f"Total number of questions: {len(ground_truth['rag_questions'])}")
        questions_with_sources = sum(1 for q in ground_truth['rag_questions'] if 'sources' in q)
        print(f"Total number of questions with sources: {questions_with_sources}")
        student_with_sources = sum(1 for q in student_results['search_results'] if 'retrieved_sources' in q)
        print(f"Total number of questions with student sources: {student_with_sources}")
        print("")
        print("Evaluation Results")
        print("========================================", end="\n")

        student_map = {
            result['question_id']: result['retrieved_sources']
            for result in student_results['search_results']
        }

        for k in [1, 3, 5, 10]:
            recall_sum = 0
            num_questions = 0

            for question in ground_truth['rag_questions']:
                if not question.get('sources'):
                    continue
                qt_sources = question['sources']
                student_sources = student_map.get(question['question_id'], [])[:k]
                hits = 0
                for qt_source in qt_sources:
                    for student_source in student_sources:
                        if qt_source['file_path'] == student_source['file_path']:
                            overlap = min(qt_source['last_character_index'], student_source['last_character_index']) - max(qt_source['first_character_index'], student_source['first_character_index'])
                            union = max(qt_source['last_character_index'], student_source['last_character_index']) - min(qt_source['first_character_index'], student_source['first_character_index'])
                            if overlap / union >= 0.05:
                                hits += 1
                                break
                num_questions += 1
                recall_sum += hits / len(qt_sources)

            if k == 1:
                print(f"Questions evaluated: {num_questions}")
            print(f"Recall@{k}: {(recall_sum / num_questions):.3f} ({(recall_sum / num_questions) * 100:.1f}%)")