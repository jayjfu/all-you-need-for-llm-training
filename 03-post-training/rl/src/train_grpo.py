import os
import re
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

MODEL = "Qwen/Qwen3.5-0.8B-Base"
OUTPUT_ROOT = "../ckpts"
EPOCHS = 1
LR = 5e-6
BATCH_SIZE = 1
NUM_GENERATIONS = 4
TEMPERATURE = 0.9
TOP_P = 0.95

BETA = 0.04
MAX_NEW_TOKENS = 512
CLIP_EPS = 0.2

os.makedirs(OUTPUT_ROOT, exist_ok=True)

raw = load_dataset("qwedsacf/competition_math", split="train")
filtered = raw.filter(lambda ex: ex["level"] in {"Level 1", "Level 2", "Level 3", "Level 4"})

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

def accuracy_reward(completions, ground_truth):
    rewards = []
    for comp, gt in zip(completions, ground_truth):
        pred = extract_boxed(comp)
        rewards.append(1.0 if pred == gt.strip() else 0.0)
    return rewards

def format_reward(completions):
    return [0.5 if "\\boxed{" in c else 0.0 for c in completions]

def compute_rewards(completions, ground_truths):
    acc = accuracy_reward(completions, ground_truths)
    fmt = format_reward(completions)
    return [a + f for a, f in zip(acc, fmt)]

tokenizer = AutoTokenizer.from_pretrained(MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(device)
model.gradient_checkpointing_enable() # trade compute for memory
ref_model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(device)
ref_model.eval()
for p in ref_model.parameters():
    p.requires_grad = False

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps = (len(dataset) // BATCH_SIZE) * EPOCHS
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=max(10, total_steps // 10),
    num_training_steps=total_steps,
)

def get_logprobs(model, input_ids, attention_mask):
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    token_logps = -F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    ).view(shift_logits.size(0), -1)
    return token_logps

def compute_advantages(rewards, num_generations, eps=1e-8):
    rewards = torch.tensor(rewards, dtype=torch.float32, device=model.device)
    grouped = rewards.view(-1, num_generations)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, keepdim=True)
    advantages = (grouped - mean) / (std + eps)
    advantages = torch.where(std < eps, torch.zeros_like(advantages), advantages)
    return advantages.view(-1)

def rollout(model, tokenizer, prompts, num_generations, max_new_tokens, temperature, top_p):
    all_completions = []
    all_full_ids = []
    all_prompt_lens = []
    all_completion_lens = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_len = inputs.input_ids.shape[1]

        for _ in range(num_generations):
            with torch.no_grad():
                gen_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            full_ids = gen_ids[0]
            completion_ids = full_ids[prompt_len:]
            completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)

            all_completions.append(completion_text)
            all_full_ids.append(full_ids)
            all_prompt_lens.append(prompt_len)
            all_completion_lens.append(len(completion_ids))

    return all_completions, all_full_ids, all_prompt_lens, all_completion_lens

def grpo_loss(token_logps, old_logps, ref_token_logps, advantages, completion_mask, beta=0.04, clip_eps=0.2):
    ratio = torch.exp(token_logps - old_logps)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)

    unclipped = ratio * advantages.unsqueeze(1)
    clipped = clipped_ratio * advantages.unsqueeze(1)
    surrogate = torch.min(unclipped, clipped)

    log_ratio_ref = ref_token_logps - token_logps
    token_kl = torch.exp(log_ratio_ref) - log_ratio_ref - 1.0

    token_loss = -(surrogate - beta * token_kl)

    seq_loss = (token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1.0)
    loss = seq_loss.mean()

    with torch.no_grad():
        mean_kl = ((token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1.0)).mean()
        mean_ratio = ((ratio * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1.0)).mean()

    return loss, mean_kl.item(), mean_ratio.item()

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

for epoch in range(EPOCHS):
    model.train()
    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

    for batch in pbar:
        prompts = batch["prompt"]
        ground_truths = batch["ground_truth"]

        all_completions, all_full_ids, all_prompt_lens, all_completion_lens = rollout(
            model, tokenizer, prompts, NUM_GENERATIONS, MAX_NEW_TOKENS, TEMPERATURE, TOP_P
        )

        expanded_gt = [gt for gt in ground_truths for _ in range(NUM_GENERATIONS)]
        rewards = compute_rewards(all_completions, expanded_gt)
        advantages = compute_advantages(rewards, NUM_GENERATIONS)

        max_len = max(len(ids) for ids in all_full_ids)
        padded_ids = torch.full(
            (len(all_full_ids), max_len),
            tokenizer.pad_token_id,
            dtype=torch.long,
            device=model.device,
        )
        attention_mask = torch.zeros_like(padded_ids)
        completion_mask = torch.zeros(len(all_full_ids), max_len - 1, dtype=torch.float32, device=model.device)

        for i, (ids, p_len, c_len) in enumerate(zip(all_full_ids, all_prompt_lens, all_completion_lens)):
            padded_ids[i, :len(ids)] = ids
            attention_mask[i, :len(ids)] = 1
            start = p_len - 1
            end = p_len - 1 + c_len
            completion_mask[i, start:end] = 1.0

        with torch.no_grad():
            old_logps = get_logprobs(model, padded_ids, attention_mask)
            ref_logps = get_logprobs(ref_model, padded_ids, attention_mask)

        for _ in range(2):
            policy_logps = get_logprobs(model, padded_ids, attention_mask)
            loss, mean_kl, mean_ratio = grpo_loss(
                policy_logps, old_logps, ref_logps, advantages, completion_mask, BETA, CLIP_EPS
            )

            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()

        mean_reward = sum(rewards) / len(rewards)
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "reward": f"{mean_reward:.3f}",
            "kl": f"{mean_kl:.4f}",
            "ratio": f"{mean_ratio:.3f}",
        })

model.save_pretrained(os.path.join(OUTPUT_ROOT, "qwen3.5-0.8b-grpo-math"))