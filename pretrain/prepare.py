"""Normalize the ABC headlines, train a small BPE vocab, and tokenize everything.

Usage: uv run python prepare.py ../data/raw/abcnews-date-text.tsv ../data
"""

import csv
import hashlib
import heapq
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

VOCAB_SIZE = 4096
CTX = 32
EOT = "<|eot|>"
SPACE = "▁"  # "▁" marks the start of a word, sentencepiece style
INF = 1 << 30
ALLOWED = re.compile(r"[^a-z0-9 '\-$%&.,:?!/]")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = ALLOWED.sub(" ", text.lower())
    return " ".join(text.split())


def read_headlines(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        sample = f.read(4096)
        f.seek(0)
        dialect = csv.excel_tab if sample.count("\t") > sample.count(",") else csv.excel
        rows = csv.reader(f, dialect)
        header = next(rows)
        col = header.index("headline_text")
        seen, out = set(), []
        for row in rows:
            if len(row) <= col:
                continue
            h = normalize(row[col])
            if h and h not in seen:
                seen.add(h)
                out.append(h)
    return out


def train_bpe(word_counts: Counter, vocab_size: int) -> tuple[list[str], list[tuple[str, str]]]:
    words = [list(w) for w in word_counts]
    counts = list(word_counts.values())
    alphabet = sorted({c for w in words for c in w})

    pair_counts: Counter = Counter()
    where: dict[tuple[str, str], set[int]] = defaultdict(set)
    for i, w in enumerate(words):
        for pair in zip(w, w[1:]):
            pair_counts[pair] += counts[i]
            where[pair].add(i)

    heap = [(-c, p) for p, c in pair_counts.items()]
    heapq.heapify(heap)
    merges = []
    while 1 + len(alphabet) + len(merges) < vocab_size and heap:
        neg, pair = heapq.heappop(heap)
        if pair_counts.get(pair, 0) != -neg:
            continue  # stale entry
        if -neg < 2:
            break
        merges.append(pair)
        merged = pair[0] + pair[1]
        touched = Counter()
        for i in list(where[pair]):
            w, c = words[i], counts[i]
            for p in zip(w, w[1:]):
                pair_counts[p] -= c
                touched[p] += 0
            out, j = [], 0
            while j < len(w):
                if j + 1 < len(w) and w[j] == pair[0] and w[j + 1] == pair[1]:
                    out.append(merged)
                    j += 2
                else:
                    out.append(w[j])
                    j += 1
            words[i] = out
            for p in zip(out, out[1:]):
                pair_counts[p] += c
                where[p].add(i)
                touched[p] += 0
        del where[pair]
        pair_counts.pop(pair, None)
        for p in touched:
            if pair_counts.get(p, 0) > 0:
                heapq.heappush(heap, (-pair_counts[p], p))
    return alphabet, merges


class Tokenizer:
    def __init__(self, vocab: list[str], merges: list[tuple[str, str]]):
        self.vocab = vocab
        self.ids = {s: i for i, s in enumerate(vocab)}
        self.ranks = {p: r for r, p in enumerate(merges)}
        self.cache: dict[str, list[int]] = {}

    def encode_word(self, word: str) -> list[int]:
        if word in self.cache:
            return self.cache[word]
        syms = [c for c in word if c in self.ids]
        while len(syms) > 1:
            k = min(range(len(syms) - 1), key=lambda k: self.ranks.get((syms[k], syms[k + 1]), INF))
            if (syms[k], syms[k + 1]) not in self.ranks:
                break
            a, b = syms[k], syms[k + 1]
            out, j = [], 0
            while j < len(syms):
                if j + 1 < len(syms) and syms[j] == a and syms[j + 1] == b:
                    out.append(a + b)
                    j += 2
                else:
                    out.append(syms[j])
                    j += 1
            syms = out
        ids = [self.ids[s] for s in syms]
        self.cache[word] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        return [i for w in normalize(text).split() for i in self.encode_word(SPACE + w)]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.vocab[i] for i in ids if i != 0).replace(SPACE, " ").strip()


def main() -> None:
    src, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    headlines = read_headlines(src)
    print(f"{len(headlines):,} unique headlines")

    word_counts = Counter(SPACE + w for h in headlines for w in h.split())
    alphabet, merges = train_bpe(word_counts, VOCAB_SIZE)
    vocab = [EOT] + alphabet + [a + b for a, b in merges]
    print(f"vocab {len(vocab)} ({len(alphabet)} base symbols, {len(merges)} merges)")
    tok = Tokenizer(vocab, merges)

    seqs = [[0] + tok.encode(h) + [0] for h in headlines]
    lengths = np.array([len(s) for s in seqs])
    print("seq length p50/p99/p99.9/max:", *np.percentile(lengths, [50, 99, 99.9, 100]).astype(int))
    keep = lengths <= CTX
    print(f"dropping {(~keep).sum()} headlines longer than {CTX} tokens")

    arr = np.zeros((int(keep.sum()), CTX), dtype=np.uint16)
    is_val = []
    for row, (s, h) in enumerate((s, h) for s, h, k in zip(seqs, headlines, keep) if k):
        arr[row, : len(s)] = s
        is_val.append(hashlib.md5(h.encode()).digest()[0] < 3)  # ~1.2% held out
    is_val = np.array(is_val)
    np.save(out_dir / "train.npy", arr[~is_val])
    np.save(out_dir / "val.npy", arr[is_val])
    (out_dir / "tokenizer.json").write_text(json.dumps({"vocab": vocab, "merges": merges, "ctx": CTX}))
    n_tok = int((arr > 0).sum())
    print(f"train {int((~is_val).sum()):,} / val {int(is_val.sum()):,} headlines, {n_tok:,} content tokens")
    for h in headlines[:3]:
        ids = tok.encode(h)
        print(f"  {h!r} -> {len(ids)} tokens -> {tok.decode(ids)!r}")


if __name__ == "__main__":
    main()
