// BPE tokenizer mirroring pretrain/prepare.py: lowercase ASCII, "▁" word prefix, merges by rank.

const SPACE = "▁";

export function normalize(text) {
  return text
    .normalize("NFKD")
    .replace(/[^\x00-\x7f]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9 '\-$%&.,:?!/]/g, " ")
    .split(/\s+/)
    .filter(Boolean)
    .join(" ");
}

export class Tokenizer {
  constructor({ vocab, merges }) {
    this.vocab = vocab;
    this.ids = new Map(vocab.map((s, i) => [s, i]));
    this.ranks = new Map(merges.map(([a, b], r) => [a + "\u0000" + b, r]));
    this.cache = new Map();
    this.eot = 0;
  }

  encodeWord(word) {
    const hit = this.cache.get(word);
    if (hit) return hit;
    let syms = [...word].filter((c) => this.ids.has(c));
    while (syms.length > 1) {
      let best = -1, bestRank = Infinity;
      for (let k = 0; k < syms.length - 1; k++) {
        const r = this.ranks.get(syms[k] + "\u0000" + syms[k + 1]);
        if (r !== undefined && r < bestRank) { bestRank = r; best = k; }
      }
      if (best < 0) break;
      const [a, b] = [syms[best], syms[best + 1]];
      const out = [];
      for (let j = 0; j < syms.length; j++) {
        if (j + 1 < syms.length && syms[j] === a && syms[j + 1] === b) { out.push(a + b); j++; }
        else out.push(syms[j]);
      }
      syms = out;
    }
    const ids = syms.map((s) => this.ids.get(s));
    this.cache.set(word, ids);
    return ids;
  }

  encode(text) {
    return normalize(text).split(" ").filter(Boolean).flatMap((w) => this.encodeWord(SPACE + w));
  }

  decode(ids) {
    return ids.filter((i) => i !== this.eot).map((i) => this.vocab[i]).join("").replaceAll(SPACE, " ").trim();
  }
}
