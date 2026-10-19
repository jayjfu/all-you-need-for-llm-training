import argparse
import re

from datasets import load_dataset
from vllm import LLM, SamplingParams


class Task:
    @property
    def eval_type(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

    def evaluate(self, conversation, response):
        raise NotImplementedError


class GSM8KTask(Task):
    GSM_pattern = r"#### (\-?[0-9\.\,]+)" # extract answer after #### marker

    def __init__(self, split="test", max_examples=None):
        ds = load_dataset("openai/gsm8k", "main", split=split)
        if max_examples is not None:
            ds = ds.select(range(min(max_examples, len(ds))))
        self.ds = ds

    @property
    def eval_type(self):
        return "generative"

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        question = row["question"]
        match = re.search(self.GSM_pattern, row["answer"])
        answer = match.group(1).strip().replace(",", "") if match else ""
        return {
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": f"#### {answer}"},
            ],
            "answer": answer,
        }

    def evaluate(self, conversation, response):
        numbers = re.findall(r"-?[0-9.,]+", response)
        if numbers:
            pred = numbers[-1].replace(",", "").strip()
            return pred == conversation["answer"]
        return False


class MMLUTask(Task):
    LETTERS = ["A", "B", "C", "D"]

    def __init__(self, subject="all", split="test", max_examples=None):
        ds = load_dataset("cais/mmlu", subject, split=split)
        if max_examples is not None:
            ds = ds.select(range(min(max_examples, len(ds))))
        self.ds = ds

    @property
    def eval_type(self):
        return "categorical"

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        question = row["question"]
        choices = row["choices"]
        answer_idx = row["answer"]
        answer_letter = self.LETTERS[answer_idx]

        prompt = f"Multiple Choice question: {question}\n"
        for letter, choice in zip(self.LETTERS, choices):
            prompt += f"- {choice}={letter}\n"
        prompt += "\nRespond only with the letter of the correct answer."

        return {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer_letter},
            ],
            "letters": self.LETTERS,
            "answer": answer_letter,
        }

    def evaluate(self, conversation, response):
        cleaned = response.strip().upper() # avoid space before the letter
        for letter in self.LETTERS:
            if cleaned.startswith(letter): # look only at the first letter
                return letter == conversation["answer"]
        return False


class VLLMEngine:
    def __init__(self, model_name):
        self.llm = LLM(model=model_name, dtype="bfloat16")
        self.tokenizer = self.llm.get_tokenizer()

    def generate(self, prompt, temperature, max_new_tokens):
        sampling_params = SamplingParams(max_tokens=max_new_tokens, temperature=temperature)
        outputs = self.llm.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text


def run_eval(task, engine, max_new_tokens, temperature, max_problems):
    n = len(task) if max_problems is None else min(len(task), max_problems)
    correct = 0

    for i in range(n):
        conv = task[i]
        user_msg = conv["messages"][0]["content"]
        prompt = f"User: {user_msg}\nAssistant:"

        gen_tokens = 8 if task.eval_type == "categorical" else max_new_tokens # short completion for categorical tasks
        response = engine.generate(prompt, max_new_tokens=gen_tokens, temperature=temperature)

        is_correct = task.evaluate(conv, response)
        correct += int(is_correct)

        status = "✓" if is_correct else "✗"
        if len(response) > 200:
            preview = response[:50].replace("\n", " ") + " ... " + response[-100:].replace("\n", " ")
        else:
            preview = response.replace("\n", " ")
        print(f"[{i+1}/{n}] {status}  {preview}")
        if task.eval_type == "generative":
            print(f"  GT: {conv.get('answer', 'N/A')}")

    acc = correct / n
    print(f"\nAccuracy: {correct}/{n} = {100*acc:.2f}%")
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="gsm8k", choices=["gsm8k", "mmlu"])
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-problems", type=int, default=None)
    args = parser.parse_args()

    if args.task == "gsm8k":
        task = GSM8KTask(max_examples=args.max_problems)
    elif args.task == "mmlu":
        task = MMLUTask(max_examples=args.max_problems)
    else:
        raise ValueError(f"Unsupported task: {args.task}")

    engine = VLLMEngine(args.model)

    run_eval(
        task,
        engine,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        max_problems=args.max_problems,
    )


if __name__ == "__main__":
    main()