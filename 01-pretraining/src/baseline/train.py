import argparse
import json
import time
import math
from pathlib import Path

import torch

from model import GPT, GPTConfig
from schedule import cosine_schedule, wsd_schedule
from tokenizer import BPETokenizer
from data import prepare_slimpajama, DataReader

ROOT = Path(__file__).resolve().parents[3]


def param_groups(model, weight_decay):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2 and not name.endswith("wte.weight"):
            decay.append(param)
        else:
            no_decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


@torch.no_grad()
def estimate_loss(model, get_batch, eval_batches):
    out = {}
    for split in ("train", "val"):
        losses, correct, count = [], 0, 0
        for _ in range(eval_batches):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses.append(loss.item())
            correct += (logits.argmax(-1) == y).sum().item()
            count += y.numel()
        mean = sum(losses) / len(losses)
        out[f"{split}_loss"] = mean
        out[f"{split}_ppl"] = math.exp(mean)
        out[f"{split}_acc"] = correct / count
    return out


def save_checkpoint(path, model, opt, step, args, readers):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "itr": step,
            "train_step": readers["train"].step if readers else None,
            "rng": torch.get_rng_state(),
        },
        path,
    )


def load_checkpoint(path, model, opt, readers, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["optimizer"])
    torch.set_rng_state(ckpt["rng"].cpu())
    if readers and ckpt.get("train_step") is not None:
        readers["train"].set_step(ckpt["train_step"])
    return int(ckpt["itr"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vocab_size", type=int, default=1024)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_embd", type=int, default=128)
    p.add_argument("--sequence_length", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--acc_steps", type=int, default=1)
    p.add_argument("--iterations", type=int, default=1_000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--scheduler", choices=["wsd", "cos"], default="wsd")
    p.add_argument("--wsd_fract_decay", type=float, default=0.1)
    p.add_argument("--wsd_final_lr_scale", type=float, default=0.0)
    p.add_argument("--decay_type", choices=["linear", "cosine", "exp"], default="linear")
    p.add_argument("--init_div_factor", type=float, default=100.0)
    p.add_argument("--ckpt_interval", type=int, default=200)
    p.add_argument("--resume_from", type=str, default="")
    p.add_argument("--warmup_steps", type=int, default=50)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--eval_interval", type=int, default=100)
    p.add_argument("--eval_batches", type=int, default=20)
    p.add_argument("--with_replacement", action="store_true")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data_seed", type=int, default=0)
    p.add_argument("--data", type=str, default=str(ROOT / "01-pretraining/data"))
    p.add_argument("--out_dir", type=str, default=str(ROOT / "01-pretraining/src/baseline"))
    args = p.parse_args()

    torch.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)  # for multi-GPU

    sources = prepare_slimpajama(args.data, vocab_size=args.vocab_size)
    tok = BPETokenizer.load(sources["tokenizer"])
    readers = {
        split: DataReader(
            sources[split],
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            seed=args.data_seed,
            with_replacement=args.with_replacement,
        )
        for split in ("train", "val")
    }
    vocab_size = tok.vocab_size
    print(f"train tokens={readers['train'].num_tokens:,},  val tokens={readers['val'].num_tokens:,}")

    def get_batch(split):
        x, y = readers[split].sample_batch()
        return x.to(args.device), y.to(args.device)

    config = GPTConfig(
        vocab_size=vocab_size,
        sequence_length=args.sequence_length,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        bias=False,
    )
    model = GPT(config).to(args.device)
    tied = model.wte.weight.numel()
    print(
        f"params={model.num_params()/1e6:.3f}M,  vocab={vocab_size}, "
        f"embed share={tied / model.num_params():.0%},  device={args.device}"
    )

    groups = param_groups(model, args.weight_decay)
    for group in groups:
        group["base_lr"] = args.lr
    opt = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.95))
    print("opt=adamw")
    if args.scheduler == "wsd":
        schedule = wsd_schedule(
            args.iterations,
            final_lr_factor=args.wsd_final_lr_scale,
            n_warmup=args.warmup_steps,
            init_div_factor=args.init_div_factor,
            fract_decay=args.wsd_fract_decay,
            decay_type=args.decay_type,
        )
    else:
        schedule = cosine_schedule(args.iterations, args.warmup_steps, min_factor=args.wsd_final_lr_scale or 0.1)

    out_dir = Path(args.out_dir)
    latest = out_dir / "ckpts" / "latest" / "base.pt"
    log_path = out_dir / "log.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)

    start_step = 0
    if args.resume_from:
        resume_path = Path(args.resume_from)
        start_step = load_checkpoint(resume_path, model, opt, readers, args.device)
        print(f"resumed from {resume_path} at step {start_step}")

    model.train()
    t0 = time.time()
    for step in range(start_step, args.iterations):
        factor = schedule(step)
        for group in opt.param_groups:
            group["lr"] = group["base_lr"] * factor
        lr = opt.param_groups[0]["lr"]

        opt.zero_grad(set_to_none=True)
        for _ in range(args.acc_steps):
            x, y = get_batch("train")
            _, loss = model(x, y)
            (loss / args.acc_steps).backward()
        grad_norm = 0.0
        if args.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item()
        opt.step()

        done = step + 1
        if done % args.ckpt_interval == 0 or done == args.iterations:
            save_checkpoint(latest, model, opt, done, args, readers)

        if step % args.eval_interval == 0 or step == args.iterations - 1:
            stats = estimate_loss(model, get_batch, args.eval_batches)
            dt = time.time() - t0
            tokens = (step - start_step + 1) * args.batch_size * args.acc_steps * args.sequence_length
            print(
                f"step={step:<5d} | val_ppl: {stats['val_ppl']:.1f} | val_loss: {stats['val_loss']:.4f} | "
                f"train_loss:{stats['train_loss']:.4f} | acc:{stats['val_acc']:.3f} | "
                f"lr:{lr:.2e} | gn:{grad_norm:.2f} | tok/s:{tokens / dt:,.0f}"
            )
            with log_path.open("a") as f:
                f.write(json.dumps({"step": step, "lr": lr, "grad_norm": grad_norm, **stats}) + "\n")

    print(f"saved {latest}")


if __name__ == "__main__":
    main()