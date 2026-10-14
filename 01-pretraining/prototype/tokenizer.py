import json
import re
from pathlib import Path
from collections import Counter

# GPT-2 pretokenizer
GPT2_PAT = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+"""
)

SPECIAL_TOKENS = [
    "<|bos|>",
    "<|user_start|>",
    "<|user_end|>",
    "<|assistant_start|>",
    "<|assistant_end|>",
    "<|python_start|>",
    "<|python_end|>",
    "<|output_start|>",
    "<|output_end|>",
]


def _pair_counts(words):
    counts = Counter()
    for word, freq in words.items():
        for i in range(len(word) - 1):
            counts[(word[i], word[i + 1])] += freq
    return counts


def _merge(words, pair, new_id):
    merged = {}
    a, b = pair
    for word, freq in words.items():
        out = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                out.append(new_id)
                i += 2
            else:
                out.append(word[i])
                i += 1
        key = tuple(out)
        merged[key] = merged.get(key, 0) + freq
    return merged


class BPETokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.special_to_id = {}
        self.id_to_special = {}

    @property
    def vocab_size(self):
        return len(self.vocab) + len(self.special_to_id)

    def bos_id(self):
        return self.special_to_id["<|bos|>"]

    def train(self, text, vocab_size, verbose=True):
        n_special = len(SPECIAL_TOKENS)
        assert vocab_size >= 256 + n_special, f"vocab_size must be >= {256 + n_special}"
        n_merges = vocab_size - 256 - n_special

        words = Counter()
        for chunk in GPT2_PAT.findall(text):
            words[tuple(chunk.encode("utf-8"))] += 1

        merges = {}
        vocab = {i: bytes([i]) for i in range(256)}
        for i in range(n_merges):
            counts = _pair_counts(words)
            if not counts:
                break
            pair = max(counts, key=counts.get)
            new_id = 256 + i
            words = _merge(words, pair, new_id)
            merges[pair] = new_id
            vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
            if verbose and (i < 5 or (i + 1) % 100 == 0 or i + 1 == n_merges):
                print(f"merge {i + 1:4d}/{n_merges} | {str(pair):<10} -> {new_id:4d} | freq: {counts[pair]:,}")

        self.merges = merges
        self.vocab = vocab
        self.special_to_id = {tok: 256 + n_merges + i for i, tok in enumerate(SPECIAL_TOKENS)}
        self.id_to_special = {i: tok for tok, i in self.special_to_id.items()}

    def _encode_word(self, word):
        ids = list(word)
        while len(ids) >= 2:
            pairs = [(self.merges.get((ids[i], ids[i + 1]), 10**18), i) for i in range(len(ids) - 1)]
            rank, pos = min(pairs)
            if rank == 10**18:
                break
            ids = ids[:pos] + [rank] + ids[pos + 2:]
        return ids

    def encode(self, text, add_bos=False):
        special_pat = re.compile("(" + "|".join(re.escape(t) for t in SPECIAL_TOKENS) + ")") # keep atomic
        ids = []
        for part in special_pat.split(text):
            if part in self.special_to_id:
                ids.append(self.special_to_id[part])
                continue
            for chunk in GPT2_PAT.findall(part):
                ids.extend(self._encode_word(tuple(chunk.encode("utf-8"))))
        if add_bos:
            ids = [self.bos_id()] + ids
        return ids

    def decode(self, ids, skip_special=False):
        out = bytearray()
        for i in ids:
            if i in self.id_to_special:
                if not skip_special:
                    out.extend(self.id_to_special[i].encode("utf-8"))
                continue
            out.extend(self.vocab[i])
        return out.decode("utf-8", errors="replace")

    def bytes_per_token(self, text):
        ids = self.encode(text, add_bos=False)
        if not ids:
            return 0.0
        return len(text.encode("utf-8")) / len(ids)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "merges": [{"a": a, "b": b, "id": i} for (a, b), i in self.merges.items()],
            "special_tokens": self.special_to_id,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "BPETokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls()
        merges = {}
        vocab = {i: bytes([i]) for i in range(256)}
        for item in payload["merges"]:
            pair = (item["a"], item["b"])
            merges[pair] = item["id"]
            vocab[item["id"]] = vocab[item["a"]] + vocab[item["b"]]
        tok.merges = merges
        tok.vocab = vocab
        tok.special_to_id = {k: int(v) for k, v in payload["special_tokens"].items()}
        tok.id_to_special = {i: t for t, i in tok.special_to_id.items()}
        return tok