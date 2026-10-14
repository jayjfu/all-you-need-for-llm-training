import argparse
import time
from pathlib import Path
from dataclasses import asdict

import torch

from model import GPT, config_from_depth
from tokenizer import BPETokenizer
from render import render, batch_rows, prompt_ids

ROOT = Path(__file__).resolve().parents[2]

IDENTITY = [
    [{"role": "user", "content": "Who are you?"}, {"role": "assistant", "content": "I am nanogpt."}],
    [{"role": "user", "content": "What is your name?"}, {"role": "assistant", "content": "My name is nanogpt."}],
    [{"role": "user", "content": "Who made you?"}, {"role": "assistant", "content": "I was trained in this repo."}],
    [{"role": "user", "content": "Say hello."}, {"role": "assistant", "content": "Hello."}],
]

CHAT = [
    [{"role": "user", "content": "What is 2 plus 2?"}, {"role": "assistant", "content": "4."}],
    [{"role": "user", "content": "Name a color."}, {"role": "assistant", "content": "Blue."}],
    [{"role": "user", "content": "Is the sun a star?"}, {"role": "assistant", "content": "Yes."}],
    [{"role": "user", "content": "Name a city."}, {"role": "assistant", "content": "Paris."}],
]

MULTIPLE_CHOICE = [
    [{"role": "user", "content": "Capital of France? A) Paris B) Rome"}, {"role": "assistant", "content": "A"}],
    [{"role": "user", "content": "2+2? A) 3 B) 4"}, {"role": "assistant", "content": "B"}],
    [{"role": "user", "content": "Water freezes at? A) 0C B) 100C"}, {"role": "assistant", "content": "A"}],
    [{"role": "user", "content": "Sun is a? A) planet B) star"}, {"role": "assistant", "content": "B"}],
]


def load_base(tokenizer, depth, seq_len, base, device):
    config = config_from_depth(depth, tokenizer.vocab_size, seq_len, head_dim=64)
    model = GPT(config)
    ckpt = torch.load(base, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    return model.to(device)


def rendered_rows(tokenizer, seq_len):
    rendered = [render(m, tokenizer) for m in IDENTITY + CHAT + MULTIPLE_CHOICE]
    rows = batch_rows(rendered, seq_len, tokenizer.bos_id())
    return rows


@torch.no_grad()
def reply(model, tokenizer, prompt, device, max_new=64):
    ids = prompt_ids([{"role": "user", "content": prompt}], tokenizer)
    input_ids = torch.tensor([ids], device=device)
    end = tokenizer.special_to_id["<|assistant_end|>"]
    for _ in range(max_new):
        logits, _ = model(input_ids[:, -model.config.sequence_len:])
        next_id = int(logits[0, -1].argmax())
        input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=device)], dim=1)
        if next_id == end:
            break
    text = tokenizer.decode(input_ids[0, len(ids):].tolist(), skip_special=True)
    return text


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=48)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--tokenizer", type=str, default=str(ROOT / "01-pretraining/data/prototype/tinyshakespeare/tokenizer.json"))
    p.add_argument("--base", type=str, default=str(ROOT / "01-pretraining/prototype/checkpoints/base.pt"))
    p.add_argument("--out", type=str, default=str(ROOT / "03-post-training/prototype/checkpoints"))
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    tokenizer = BPETokenizer.load(args.tokenizer)
    rows = rendered_rows(tokenizer, args.seq_len)
    print(f"rows={len(rows)}, seq={args.seq_len}")

    model = load_base(tokenizer, args.depth, args.seq_len, Path(args.base), device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))

    model.train()
    t0 = time.time()
    for step in range(args.steps):
        ids, mask = rows[step % len(rows)]
        x = torch.tensor([ids[:-1]], device=device)
        y = torch.tensor([ids[1:]], device=device)
        m = torch.tensor([mask[1:]], dtype=torch.float32, device=device)
        _, loss = model(x, y, m)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 20 == 0 or step == args.steps - 1:
            print(f"step={step:<4} | loss: {loss.item():.4f}")

    for q in ["Who are you?", "What is 2 plus 2?", "Capital of France? A) Paris B) Rome"]:
        print(f"Q: {q}")
        print(f"A: {reply(model, tokenizer, q, device)!r}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": asdict(model.config)}, out / "model.pt")
    print(f"saved {out / "model.pt"}, elapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()