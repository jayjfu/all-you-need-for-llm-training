import os
import re

from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset

MODEL = "Qwen/Qwen3.5-0.8B-Base"
EPOCHS = 1
OUTPUT_ROOT = "../ckpts"

raw = load_dataset("qwedsacf/competition_math", split="train")

def is_level_1_to_4(ex):
    return ex["level"] in {"Level 1", "Level 2", "Level 3", "Level 4"}

filtered = raw.filter(is_level_1_to_4)

def extract_boxed(text):
    m = re.search(r"\\boxed\{(.*?)\}", text)
    return m.group(1).strip() if m else ""

def make_prompt(ex):
    return {
        "prompt": (
            "Solve the following math competition problem step by step. "
            "Put the final answer inside \\boxed{}.\n\n"
            f"{ex['problem']}"
        ),
        "ground_truth": extract_boxed(ex["solution"]),
    }

dataset = filtered.map(make_prompt, remove_columns=filtered.column_names)

def accuracy_reward(completions, ground_truth, **kwargs):
    rewards = []
    for comp, gt in zip(completions, ground_truth):
        pred = extract_boxed(comp)
        rewards.append(1.0 if pred == gt.strip() else 0.0)
    return rewards

def format_reward(completions, **kwargs):
    return [0.5 if "\\boxed{" in c else 0.0 for c in completions]

out_dir = os.path.join(OUTPUT_ROOT, "qwen3.5-0.8b-grpo-math")

args = GRPOConfig(
    output_dir=out_dir,
    num_train_epochs=EPOCHS, # max_steps
    learning_rate=5e-6,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    bf16=True,
    gradient_checkpointing=True,
    num_generations=4,
    temperature=0.9,
    beta=0.04,
)

trainer = GRPOTrainer(
    model=MODEL,
    reward_funcs=[accuracy_reward, format_reward],
    args=args,
    train_dataset=dataset,
)

trainer.train()
trainer.save_model(os.path.join(out_dir, "final"))