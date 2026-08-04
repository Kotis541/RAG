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

    def index(self, max_chunk_size: int = 2000):
        """Index the repository."""
        try:
            chunker = RagChunker()
            parser = RagParser()
            docs = parser.load_vocabulary("data/raw/vllm-0.10.1")
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
        for result in top_result:
            print(f"{result['file_path']} [{result['first_character_index']}:{result['last_character_index']}]")

    
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
        return f"Saved student_search_results saved to {save_directory}"


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

            final_output = StudentSearchResultsAndAnswer(search_results=answered_results, k=k)
            with open(save_directory, 'w', encoding='utf-8') as f:
                f.write(final_output.model_dump_json(indent=2))
            print(f"  [{i+1}/{len(questions)}] Saved to {save_directory}")

        return f"Answered dataset saved to {save_directory}"


    def evaluate(self, student_answer_path: str, dataset_path: str, k: int = 10, max_context_length: int = 2000) -> str:
        with open(student_answer_path, 'r') as f:
            student_results = json.load(f)

        with open(dataset_path, 'r') as f:
            ground_truth = json.load(f)

        if StudentSearchResults.validate(student_results):
            print("Student answers is valid: True")
        print(f"Total number of questions: {len(ground_truth['rag_questions']) + 1}")
        questions_with_sources = sum(1 for q in ground_truth['rag_questions'] if 'sources' in q)
        print(f"Total number of questions with sources: {questions_with_sources + 1}")
        student_with_sources = sum(1 for q in student_results['search_results'] if 'retrieved_sources' in q)
        print(f"Total number of questions with student sources: {student_with_sources + 1}")
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
                if not question['sources']:
                    continue
                qt_sources = question['sources']
                student_sources = student_map.get(question['question_id'], [])[:k]
                hits = 0
                for qt_source in qt_sources:
                    for student_source in student_sources:
                        if qt_source['file_path'] == student_source['file_path']:
                            overlap = min(qt_source['last_character_index'], student_source['last_character_index']) - max(qt_source['first_character_index'], student_source['first_character_index'])
                            union = max(qt_source['last_character_index'], student_source['last_character_index']) - min(qt_source['first_character_index'], student_source['first_character_index'])
                            IoU = overlap / union
                            if IoU >= 0.05:
                                hits += 1
                                break
                num_questions += 1
                recall_q = hits / len(qt_sources)
                recall_sum += recall_q
            print(f"Recall@{k}: {(recall_sum / num_questions):.3f}")