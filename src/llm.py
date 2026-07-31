from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class LLMGenerator:
    """A class to handle LLM model loading and answer generation."""
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B"):
        """
        Initializes the generator by loading the model and tokenizer.
        It will automatically use a GPU if available.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        # Left-padding is required for batch generation with causal LMs
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _build_prompt(self, context: str, user_question: str) -> str:
        """Build a chat prompt from context and question."""
        messages = [
            {"role": "system", "content": "Directly answer the user's question in a single, concise sentence based on the provided context. Do not include your thinking process."},
            {"role": "user", "content": f"Context: {context}\nQuestion: {user_question}"}
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def generate_answer(self, context: str, user_question: str) -> str:
        """
        Generates an answer based on the provided context and question.
        """
        prompt = self._build_prompt(context, user_question)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        outputs = self.model.generate(**inputs, max_new_tokens=60, do_sample=False)
        new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        final_answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        return final_answer

    def generate_answer_batch(self, contexts: list[str], questions: list[str], batch_size: int = 8) -> list[str]:
        """
        Generates answers for multiple context-question pairs in batches.
        Much faster than calling generate_answer() in a loop.
        """
        all_prompts = [self._build_prompt(ctx, q) for ctx, q in zip(contexts, questions)]
        all_answers = []

        for i in range(0, len(all_prompts), batch_size):
            batch_prompts = all_prompts[i:i + batch_size]
            inputs = self.tokenizer(batch_prompts, return_tensors="pt", padding=True).to(self.device)
            input_length = inputs['input_ids'].shape[1]

            outputs = self.model.generate(**inputs, max_new_tokens=60, do_sample=False)

            for output in outputs:
                new_tokens = output[input_length:]
                answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                all_answers.append(answer)

            print(f"  Batch {i // batch_size + 1}/{(len(all_prompts) + batch_size - 1) // batch_size} done ({len(all_answers)}/{len(all_prompts)} questions)")

        return all_answers

