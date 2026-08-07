from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class LLMGenerator:
    """Loads a causal LM and generates single-sentence answers from a context + question pair."""

    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B"):
        """Load the tokenizer and model onto the best available device (GPU or CPU)."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.device.type == "cpu":
            num_threads = torch.get_num_threads()
            torch.set_num_threads(num_threads)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _build_prompt(self, context: str, question: str) -> str:
        """Format context and question into a chat-template prompt (context truncated to 1500 chars)."""
        context = context[:1500]
        messages = [
            {"role": "system", "content": "Directly answer the user's question in a single, concise sentence based on the provided context. Do not include your thinking process."},
            {"role": "user", "content": f"Context: {context}\nQuestion: {question}"}
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def generate_answer(self, context: str, question: str) -> str:
        """Generate a short answer for the given context and question using greedy decoding."""
        prompt = self._build_prompt(context, question)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=60, do_sample=False)

        new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
