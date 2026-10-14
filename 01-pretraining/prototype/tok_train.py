import argparse
from pathlib import Path
import urllib.request

from tokenizer import BPETokenizer

ROOT = Path(__file__).resolve().parents[2]

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def get_text(path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SHAKESPEARE_URL, path)
    return path.read_text(encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vocab-size", type=int, default=1024)
    p.add_argument("--data", type=str, default=str(ROOT / "01-pretraining/data/prototype/tinyshakespeare/input.txt"))
    p.add_argument("--out", type=str, default=str(ROOT / "01-pretraining/data/prototype/tinyshakespeare/tokenizer.json"))
    args = p.parse_args()

    data_path = Path(args.data)
    text = get_text(data_path)
    print(f"training BPE on {len(text):,} chars, vocab_size={args.vocab_size:,}")
    tok = BPETokenizer()
    tok.train(text, args.vocab_size)
    tok.save(args.out)
    print(f"saved {args.out}, vocab={tok.vocab_size:,}, bytes/token={tok.bytes_per_token(text[:20000]):.3f}")


if __name__ == "__main__":
    main()