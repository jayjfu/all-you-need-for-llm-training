import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from model import GPT, GPTConfig
from tokenizer import BPETokenizer
from render import prompt_ids

ROOT = Path(__file__).resolve().parents[2]

PAIRS = [
    ("What is 2 plus 2?", "4"),
    ("What is 5 minus 2?", "3"),
    ("What is 10 divided by 2?", "5"),
]


def load_model(path, device):
    ckpt = torch.load(path, map_location="cpu")
    model = GPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    return model.to(device)


def reply(model, tokenizer, prompt, device, max_new=64):
    ids = prompt_ids([{"role": "user", "content": prompt}], tokenizer)
    input_ids = torch.tensor([ids], device=device)
    logps = []
    for _ in range(max_new):
        logits, _ = model(input_ids[:, -model.config.sequence_len :])
        logp = F.log_softmax(logits[:, -1, :], dim=-1)
        next_id = torch.multinomial(logp.exp(), 1)
        logps.append(logp.gather(1, next_id))
        input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=device)], dim=1)
        if next_id == tokenizer.special_to_id["<|assistant_end|>"]:
            break
    text = tokenizer.decode(input_ids[0, len(ids):].tolist(), skip_special=True)
    return text, logps


def reward(text, gold):
    return 1.0 if gold.lower() in text.lower() else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", type=str, default=str(ROOT / "01-pretraining/data/prototype/tinyshakespeare/tokenizer.json"))
    p.add_argument("--ckpt", type=str, default=str(ROOT / "03-post-training/prototype/checkpoints/model.pt"))
    p.add_argument("--group_size", type=int, default=4)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BPETokenizer.load(args.tokenizer)
    model = load_model(Path(args.ckpt), device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    hits = 0
    total = 0

    for step, (q, gold) in enumerate(PAIRS * 4):
        # sample a group
        group_logps = []
        group_rewards = []
        group_texts = []

        model.train()
        for _ in range(args.group_size):
            text, logps = reply(model, tokenizer, q, device)
            r = reward(text, gold)

            group_logps.append(logps)
            group_rewards.append(r)
            group_texts.append(text)

        # group-relative advantages
        rewards = torch.tensor(group_rewards, device=device)
        mean_r = rewards.mean()
        std_r = rewards.std() + 1e-8
        advantages = (rewards - mean_r) / std_r

        # policy gradient loss over the group
        loss = 0.0
        for logps, adv in zip(group_logps, advantages):
            if len(logps) == 0:
                continue
            seq_logp = torch.cat(logps).sum()
            loss = loss - seq_logp * adv

        loss = loss / args.group_size
        loss.backward()
        opt.step()
        opt.zero_grad()

        hits += rewards.sum().item()
        total += args.group_size
        if step % 3 == 0:
            print(f"step={step:<2} | mean_r: {mean_r:.2f} | adv: {advantages.tolist()} | sample: {group_texts[0]!r}")
    print(f"mean reward: {hits / total:.2f}")


if __name__ == "__main__":
    main()