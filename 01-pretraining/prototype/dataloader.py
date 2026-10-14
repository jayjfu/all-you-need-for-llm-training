from dataclasses import dataclass

import torch


@dataclass
class PackStats:
    rows: int = 0
    docs_placed: int = 0
    tokens_cropped: int = 0
    tokens_emitted: int = 0

    @property
    def crop_ratio(self):
        total = self.tokens_emitted + self.tokens_cropped
        return self.tokens_cropped / total if total else 0.0


def tokenize_docs(docs, tokenizer):
    bos = tokenizer.bos_id()
    return [[bos] + tokenizer.encode(text) for text in docs]


# Documents packed using best-fit algorithm to minimize cropping
# When no document fits remaining space, crops a document to fill exactly
def best_fit_pack(docs, seq_len, buffer_size=32):
    stats = PackStats()
    rows = []
    buffer = []
    cursor = 0

    def refill():
        nonlocal cursor
        while len(buffer) < buffer_size and cursor < len(docs):
            buffer.append(docs[cursor])
            cursor += 1

    while True:
        refill()
        if not buffer:
            break
        row = []
        remaining = seq_len
        while remaining > 0 and buffer:
            fits = [i for i, doc in enumerate(buffer) if len(doc) <= remaining]
            if fits:
                choice = max(fits, key=lambda i: len(buffer[i]))
                doc = buffer.pop(choice)
                row.extend(doc)
                remaining -= len(doc)
                stats.docs_placed += 1
            else:
                choice = min(range(len(buffer)), key=lambda i: len(buffer[i]))
                doc = buffer.pop(choice)
                row.extend(doc[:remaining])
                stats.tokens_cropped += len(doc) - remaining
                stats.docs_placed += 1
                remaining = 0
        if len(row) < seq_len:
            break
        rows.append(row)
        stats.rows += 1
        stats.tokens_emitted += seq_len
    return rows, stats


class PackedLoader:
    def __init__(self, rows, batch_size, device):
        if len(rows) < batch_size:
            raise ValueError(f"need at least {batch_size} packed rows, got {len(rows)}")
        self.rows = rows
        self.batch_size = batch_size
        self.pos = 0
        self.device = device

    def next_batch(self):
        xs, ys = [], []
        for _ in range(self.batch_size):
            row = self.rows[self.pos % len(self.rows)]
            self.pos += 1
            xs.append(row[:-1])
            ys.append(row[1:])
        x = torch.tensor(xs, dtype=torch.long, device=self.device)
        y = torch.tensor(ys, dtype=torch.long, device=self.device)
        return x, y