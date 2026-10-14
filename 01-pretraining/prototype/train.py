import argparse
import time
import math
from pathlib import Path
from dataclasses import asdict

import torch
import torch.nn.functional as F

from model import GPT, config_from_depth
from muon import build_optimizer, split_params
from tokenizer import BPETokenizer
from dataset import prepare_shakespeare, read_shard
from dataloader import PackedLoader, best_fit_pack, tokenize_docs

ROOT = Path(__file__).resolve().parents[2]


# Learning rate schedule (linear warmup, constant, linear warmdown)
def lr_multiplier(step, total_steps, warmup=40, warmdown_ratio=0.65, final_frac=0.05):
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    warmdown_steps = int(total_steps * warmdown_ratio)
    warmdown_start = total_steps - warmdown_steps
    if step < warmdown_start:
        return 1.0
    progress = (step - warmdown_start) / max(warmdown_steps, 1)
    return 1.0 - progress * (1.0 - final_frac)


def token_nbytes_table(tokenizer):
    table = torch.zeros(tokenizer.vocab_size, dtype=torch.float32)
    for i, raw in tokenizer.vocab.items():
        table[i] = len(raw)
    return table


@torch.no_grad()
def evaluate_bpb(model, batches, nbytes, special_ids):
    special = torch.zeros(nbytes.numel(), dtype=torch.bool, device=nbytes.device)
    if special_ids:
        special[list(special_ids)] = True

    nats = torch.zeros((), device=nbytes.device)
    total_bytes = torch.zeros((), device=nbytes.device)
    for x, y in batches:
        logits, _ = model(x, y)
        per = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none")
        keep = ~special[y.reshape(-1)] # Mask out special tokens
        nats += per[keep].sum()
        total_bytes += nbytes[y.reshape(-1)][keep].sum()

    if total_bytes <= 0:
        return float("inf")
    return (nats / math.log(2) / total_bytes).item()


@torch.no_grad()
def sample(model, tokenizer, device, prompt="ROMEO:"):
    ids = tokenizer.encode(prompt, add_bos=True)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(input_ids, max_new_tokens=60, temperature=0.8, top_k=40)
    return tokenizer.decode(out[0].tolist(), skip_special=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--matrix-lr", type=float, default=0.02)
    p.add_argument("--embed-lr", type=float, default=3e-3)
    p.add_argument("--adamw-only", action="store_true")
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--tokenizer", type=str, default=str(ROOT / "01-pretraining/data/prototype/tinyshakespeare/tokenizer.json"))
    p.add_argument("--data", type=str, default=str(ROOT / "01-pretraining/data/prototype/tinyshakespeare"))
    p.add_argument("--out", type=str, default=str(ROOT / "01-pretraining/prototype/checkpoints"))
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    data_path = Path(args.data)
    tokenizer = BPETokenizer.load(args.tokenizer)

    train_path, val_path = prepare_shakespeare(data_path, data_path / "input.txt")
    train_docs = tokenize_docs(read_shard(train_path), tokenizer)
    val_docs = tokenize_docs(read_shard(val_path), tokenizer)
    train_rows, train_stats = best_fit_pack(train_docs, args.seq_len + 1)
    val_rows, val_stats = best_fit_pack(val_docs, args.seq_len + 1)
    print(
        f"train docs={len(train_docs):,}, rows={train_stats.rows:,}, crop={train_stats.crop_ratio:.2f}\n"
        f"val   docs={len(val_docs):,}, rows={val_stats.rows:,}, crop={val_stats.crop_ratio:.2f}"
    )

    train_loader = PackedLoader(train_rows, args.batch_size, device)
    val_loader = PackedLoader(val_rows, min(args.batch_size, len(val_rows)), device)
    nbytes = token_nbytes_table(tokenizer).to(device)
    special_ids = set(tokenizer.special_to_id.values())

    config = config_from_depth(args.depth, vocab_size=tokenizer.vocab_size, sequence_len=args.seq_len, head_dim=64)
    model = GPT(config).to(device)
    if device.type == "cuda":
        model = model.to(dtype=torch.bfloat16)
    print(f"depth={args.depth}, params={model.num_params() / 1e6:.2f}M, seq={args.seq_len}, vocab={tokenizer.vocab_size}")
    matrix, adam = split_params(model)
    print(f"matrix params={sum(p.numel() for p in matrix) / 1e6:.3f}M,  adamw params={sum(p.numel() for p in adam) / 1e6:.3f}M")

    if args.adamw_only:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
        optimizers = [optimizer]
        base_lrs = [args.lr]
    else:
        muon, adamw = build_optimizer(model, matrix_lr=args.matrix_lr, embed_lr=args.embed_lr)
        optimizers = [muon, adamw]
        base_lrs = [args.matrix_lr, args.embed_lr]

    model.train()
    t0 = time.time()
    for step in range(args.steps):
        mult = lr_multiplier(step, args.steps)
        for opt, base in zip(optimizers, base_lrs):
            for group in opt.param_groups:
                group["lr"] = base * mult

        x, y = train_loader.next_batch()
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for opt in optimizers:
            opt.step()
            opt.zero_grad(set_to_none=True)

        if step % 20 == 0 or step == args.steps - 1:
            dt = time.time() - t0
            tok_s = (step + 1) * args.batch_size * args.seq_len / dt
            print(f"step={step:<4d} | loss: {loss.item():.4f} | lr: {args.lr * mult:.2e} | tok/s: {tok_s:.0f}")

        if args.eval_every > 0 and (step + 1) % args.eval_every == 0:
            batches = [val_loader.next_batch() for _ in range(2)]
            val_bpb = evaluate_bpb(model, batches, nbytes, special_ids)
            print(f"  val bpb: {val_bpb:.4f}")
            print("  sample:", sample(model, tokenizer, device).replace("\n", " "))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": asdict(config), "step": args.steps}, out / "base.pt")
    print(f"saved {out / "base.pt"}, elapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()