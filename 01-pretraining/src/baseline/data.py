from pathlib import Path
from tqdm import tqdm

import torch
import numpy as np
from datasets import load_dataset

from tokenizer import BPETokenizer


def _write_memmap(path, tokens):
    mmap = np.memmap(path, dtype=np.uint16, mode="w+", shape=tokens.shape) # (can do since enc.max_token_value is < 2**16)
    mmap[:] = tokens
    mmap.flush()


def prepare_slimpajama(data_dir, vocab_size=4096, val_fraction=0.001):
    data_dir = Path(data_dir) / "slimpajama"
    data_dir.mkdir(parents=True, exist_ok=True)
    tok_path = data_dir / "tokenizer.json"
    train_path = data_dir / "train.bin"
    val_path = data_dir / "val.bin"
    if train_path.exists() and val_path.exists() and tok_path.exists():
        return {"train": train_path, "val": val_path, "tokenizer": tok_path}

    # ds = load_dataset("DKYoon/SlimPajama-6B", split="train")
    # sample_texts = "\n".join(ex["text"] for ex in ds.select(range(min(2000, len(ds))))) # on a small sample to keep it fast
    ds = load_dataset("DKYoon/SlimPajama-200k", split="train")
    sample_texts = "\n".join(ex["input"] for ex in ds.select(range(min(2000, len(ds))))) # on a small sample to keep it fast
    tok = BPETokenizer()
    tok.train(sample_texts, vocab_size, verbose=False)
    tok.save(tok_path)

    n_val = max(1, int(len(ds) * val_fraction))
    train_ds = ds.select(range(n_val, len(ds)))
    val_ds = ds.select(range(n_val))

    def encode_and_write(dset, path):
        all_ids = []
        for ex in tqdm(dset, desc="tokenizing data"):
            # ids = tok.encode(ex["text"])
            ids = tok.encode(ex["input"])
            all_ids.extend(ids)
        arr = np.asarray(all_ids, dtype=np.uint16)
        _write_memmap(path, arr)

    encode_and_write(train_ds, train_path)
    encode_and_write(val_ds, val_path)
    return {"train": train_path, "val": val_path, "tokenizer": tok_path}


class DataReader:
    def __init__(self, data_src, batch_size, sequence_length, seed=0, with_replacement=False, keep_in_ram=False):
        self.data_path = str(data_src)
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.seed = seed
        self.with_replacement = with_replacement
        self.step = 0
        self.epoch = -1
        self.order = None
        self.epoch_offset = 0
        probe = np.memmap(self.data_path, dtype=np.uint16, mode="r")
        self.num_tokens = int(len(probe))
        self.data = np.array(probe) if keep_in_ram else None
        del probe

    def _get_data(self):
        if self.data is not None:
            return self.data
        return np.memmap(self.data_path, dtype=np.uint16, mode="r")

    def __len__(self):
        return self.num_tokens - self.sequence_length - 1

    def set_step(self, step):
        self.step = step
        self.epoch = -1

    def _indices(self, step):
        max_start = len(self)
        if self.with_replacement:
            rng = np.random.default_rng(self.seed + step)
            return rng.integers(0, max_start, size=self.batch_size)

        n_seq = max(self.num_tokens // self.sequence_length, 1)
        epoch_length = max(n_seq // self.batch_size, 1)
        epoch = step // epoch_length
        if epoch != self.epoch:
            rng = np.random.default_rng(self.seed + epoch)
            self.order = rng.permutation(n_seq)
            self.epoch_offset = int(rng.integers(0, self.sequence_length))
            self.epoch = epoch
        epoch_idx = step % epoch_length
        window = self.order[epoch_idx * self.batch_size : (epoch_idx + 1) * self.batch_size]
        if len(window) < self.batch_size:
            window = np.concatenate([window, self.order[: self.batch_size - len(window)]])
        idxs = window * self.sequence_length + self.epoch_offset
        return np.clip(idxs, 0, max_start)

    def sample_batch(self):
        data = self._get_data()
        idxs = self._indices(self.step)
        self.step += 1
        xy = np.stack([np.asarray(data[i : i + self.sequence_length + 1], dtype=np.int64) for i in idxs])
        x = torch.from_numpy(xy[:, :-1]).contiguous()
        y = torch.from_numpy(xy[:, 1:]).contiguous()
        return x, y