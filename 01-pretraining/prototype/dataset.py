import json
import urllib.request

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def download_shakespeare(dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        urllib.request.urlretrieve(SHAKESPEARE_URL, dest)
    return dest


def split_documents(text, min_chars=80):
    docs = []
    for block in text.split("\n\n"):
        block = block.strip()
        if len(block) >= min_chars:
            docs.append(block)
    return docs


def write_shards(docs, out_dir, val_frac=0.05):
    out_dir.mkdir(parents=True, exist_ok=True)
    n_val = int(len(docs) * val_frac)
    splits = {"train": docs[:-n_val], "val": docs[-n_val:]}
    paths = {}
    for name, rows in splits.items():
        path = out_dir / f"{name}_000.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for text in rows:
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        paths[name] = path
    return paths["train"], paths["val"]


def read_shard(path):
    docs = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            docs.append(row["text"] if isinstance(row, dict) else row)
    return docs


def prepare_shakespeare(shard_dir, raw_path):
    raw_path = raw_path
    text = download_shakespeare(raw_path).read_text(encoding="utf-8")
    docs = split_documents(text)
    return write_shards(docs, shard_dir)