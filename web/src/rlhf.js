// Online DPO from A/B clicks. A trainable policy and a frozen copy of the pretrained model
// (the DPO reference) share one GPU device. Every click trains on the new pair plus a few
// replayed earlier pairs, and the implicit reward β·(log π − log π_ref) re-ranks a pool.
// Plain DPO drifts into degenerate text after a few dozen clicks, so the loss also includes a
// small per-token NLL on the chosen headline (as in RPO, arXiv:2404.19733), weighted by `sft`.
// Defaults come from test/simulate.mjs: with a consistent simulated preference, 60 clicks take
// the liked share of samples from ~17% to ~69% while fluency and diversity hold.
// Batches are padded to fixed shapes so the GPU plans are built once.

import { GPT } from "./gpt.js";
import { parseModel } from "./format.js";
import { Tokenizer } from "./tokenizer.js";

const sigmoid = (x) => 1 / (1 + Math.exp(-x));

export class DPOSession {
  /**
   * @param {GPUDevice} device
   * @param {ArrayBuffer} modelBytes contents of headline-gpt.bin
   */
  static fromBytes(device, modelBytes, opts = {}) {
    const { config, tokenizer, tensors, meta } = parseModel(modelBytes);
    const maxRows = opts.maxRows ?? 512;
    const policy = new GPT(device, config, tensors, { maxRows });
    const ref = new GPT(device, config, tensors, { maxRows });
    return new DPOSession(policy, ref, new Tokenizer(tokenizer), tensors, meta, opts);
  }

  constructor(policy, ref, tokenizer, pretrained, meta, { beta = 0.1, lr = 1e-4, sft = 0.05, stepsPerClick = 1, replay = 7, temperature = 0.9, topK = 50 } = {}) {
    Object.assign(this, { policy, ref, tokenizer, pretrained, meta, beta, lr, sft, stepsPerClick, replay, temperature, topK });
    this.pairs = []; // { chosen: ids, rejected: ids }
    this.refCache = new Map();
    this.ctx = policy.cfg.ctx;
    this.chunk = Math.floor(policy.maxRows / this.ctx); // sequences per forward pass
  }

  text(ids) { return this.tokenizer.decode(ids); }

  /** Sample n distinct headlines from the current policy. */
  async sample(n, { tries = 3 } = {}) {
    const seen = new Set();
    const out = [];
    for (let k = 0; k < tries && out.length < n; k++) {
      const rows = [];
      for (let left = n - out.length; left > 0; left -= this.chunk) {
        rows.push(...(await this.policy.generate({ n: Math.min(left, this.chunk), temperature: this.temperature, topK: this.topK })));
      }
      for (const ids of rows) {
        const key = this.text(ids);
        if (ids.length && !seen.has(key)) { seen.add(key); out.push(ids); }
      }
    }
    return out;
  }

  /** Two distinct fresh headlines. */
  async samplePair() {
    const [a, b] = await this.sample(2, { tries: 5 });
    return [a, b ?? (await this.sample(1))[0]];
  }

  // Rows [eot, ids…, eot, pad…]; a row's log-prob sums targets at positions 0..len (incl. closing eot).
  // Always B rows x ctx tokens (extra rows are empty and get zero gradient).
  _batch(seqs, B) {
    const T = this.ctx;
    const tokens = new Array(B * T).fill(0);
    seqs.forEach((s, b) => s.slice(0, T - 2).forEach((id, t) => (tokens[b * T + 1 + t] = id)));
    return { tokens, T, B, lens: seqs.map((s) => Math.min(s.length, T - 2)) };
  }

  async _seqLogp(model, seqs, B = this.chunk) {
    const { tokens, T, lens } = this._batch(seqs, B);
    const logp = await model.forward(tokens, B, T);
    return { sums: lens.map((len, b) => { let s = 0; for (let t = 0; t <= len; t++) s += logp[b * T + t]; return s; }), T, lens };
  }

  /** Reference log-probs never change, so compute each sequence once. */
  async _refLogp(seqs) {
    const missing = seqs.filter((s) => !this.refCache.has(s.join(",")));
    for (let i = 0; i < missing.length; i += this.chunk) {
      const chunk = missing.slice(i, i + this.chunk);
      const { sums } = await this._seqLogp(this.ref, chunk);
      chunk.forEach((s, k) => this.refCache.set(s.join(","), sums[k]));
    }
    return seqs.map((s) => this.refCache.get(s.join(",")));
  }

  /** Implicit DPO reward β·(log π(y) − log π_ref(y)) for each sequence. */
  async rewards(seqs) {
    const refs = await this._refLogp(seqs);
    const out = [];
    for (let i = 0; i < seqs.length; i += this.chunk) {
      const { sums } = await this._seqLogp(this.policy, seqs.slice(i, i + this.chunk));
      sums.forEach((s, k) => out.push(this.beta * (s - refs[i + k])));
    }
    return out;
  }

  /**
   * Record a preference and train on it (plus replayed pairs).
   * Returns stats from the final step: { loss, margin (of the new pair), accuracy over replayed batch, ms }.
   */
  async prefer(chosen, rejected) {
    const t0 = performance.now();
    this.pairs.push({ chosen, rejected });
    let stats;
    for (let step = 0; step < this.stepsPerClick; step++) {
      const batch = [this.pairs.at(-1), ...this._replay()];
      const seqs = batch.flatMap((p) => [p.chosen, p.rejected]);
      const refs = await this._refLogp(seqs);
      const B = 2 * (1 + this.replay);
      const { sums, T, lens } = await this._seqLogp(this.policy, seqs, B);
      const g = new Float32Array(B * T);
      let loss = 0, correct = 0;
      batch.forEach((_, i) => {
        const z = this.beta * ((sums[2 * i] - refs[2 * i]) - (sums[2 * i + 1] - refs[2 * i + 1]));
        loss += -Math.log(sigmoid(z) + 1e-12);
        correct += z > 0;
        const coef = (this.beta * sigmoid(-z)) / batch.length; // dL/dlogp = ∓coef
        for (const [row, sign] of [[2 * i, -1], [2 * i + 1, 1]]) {
          for (let t = 0; t <= lens[row]; t++) g[row * T + t] = sign * coef;
        }
        const nll = this.sft / (lens[2 * i] + 1) / batch.length; // mean NLL per token of the chosen
        for (let t = 0; t <= lens[2 * i]; t++) g[2 * i * T + t] -= nll;
        if (i === 0) stats = { margin: z };
      });
      await this.policy.backward(g, { lr: this.lr, weightDecay: 0 });
      Object.assign(stats, { loss: loss / batch.length, accuracy: correct / batch.length });
    }
    stats.ms = performance.now() - t0;
    stats.pairs = this.pairs.length;
    return stats;
  }

  _replay() {
    const old = this.pairs.slice(0, -1);
    for (let i = old.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [old[i], old[j]] = [old[j], old[i]];
    }
    return old.slice(0, this.replay);
  }

  /** Forget all preferences and go back to the pretrained weights. */
  reset() {
    this.policy.setWeights(this.pretrained);
    this.pairs = [];
  }
}
