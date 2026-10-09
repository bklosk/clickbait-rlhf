// Raw-WebGPU GPT: forward, backward and AdamW, mirroring pretrain/model.py.
//
// Every distinct (batch, length, mode) gets a "plan": the full list of dispatches with their
// bind groups and uniforms built once, so a training step is one writeBuffer + one submit.

import * as K from "./kernels.js";

const ALIGN = 64; // floats; keeps every tensor's byte offset a multiple of 256 for bind offsets
const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
const SCRATCH_FLOATS = 1 << 22; // split-K partial sums
const TARGET_WORKGROUPS = 128; // enough to fill an M-series / laptop GPU

function grid(workgroups) {
  return workgroups <= 65535 ? [workgroups, 1] : [65535, Math.ceil(workgroups / 65535)];
}

// Parameter order matters: everything before `decayEnd` gets weight decay (the 2D tensors).
function paramSpecs(cfg) {
  const { vocab: V, ctx: T, d: C, layers: L } = cfg;
  const mats = [["wte.weight", V * C], ["wpe.weight", T * C]];
  const vecs = [];
  for (let l = 0; l < L; l++) {
    const p = `blocks.${l}.`;
    mats.push(
      [p + "qkv.weight", 3 * C * C], [p + "proj.weight", C * C],
      [p + "fc.weight", 4 * C * C], [p + "fc_proj.weight", 4 * C * C],
    );
    vecs.push(
      [p + "ln1.weight", C], [p + "ln1.bias", C], [p + "qkv.bias", 3 * C], [p + "proj.bias", C],
      [p + "ln2.weight", C], [p + "ln2.bias", C], [p + "fc.bias", 4 * C], [p + "fc_proj.bias", C],
    );
  }
  vecs.push(["lnf.weight", C], ["lnf.bias", C]);
  let off = 0;
  const layout = {};
  const place = ([name, n]) => { layout[name] = { off, n }; off += Math.ceil(n / ALIGN) * ALIGN; };
  mats.forEach(place);
  const decayEnd = off;
  vecs.forEach(place);
  return { layout, total: off, decayEnd };
}

export class GPT {
  /**
   * @param {GPUDevice} device
   * @param {{vocab:number, ctx:number, d:number, layers:number, heads:number}} cfg
   * @param {Record<string, Float32Array>} tensors torch state_dict names -> fp32 data
   * @param {{maxRows?: number}} opts maxRows = largest batch*length you will run
   */
  constructor(device, cfg, tensors, { maxRows = 2048 } = {}) {
    this.device = device;
    this.cfg = cfg;
    this.maxRows = maxRows;
    const { layout, total, decayEnd } = paramSpecs(cfg);
    this.layout = layout;
    this.nParams = total;
    this.decayEnd = decayEnd;

    const flat = new Float32Array(total);
    for (const [name, { off, n }] of Object.entries(layout)) {
      const t = tensors[name];
      if (!t || t.length !== n) throw new Error(`bad tensor ${name}: ${t?.length} != ${n}`);
      flat.set(t, off);
    }
    this.params = this._buf(total * 4, flat);
    this.grads = this._buf(total * 4);
    this.adamM = this._buf(total * 4);
    this.adamV = this._buf(total * 4);
    this.adamT = 0;

    if (cfg.ctx > 32 || cfg.d / cfg.heads > 32) throw new Error("attention kernels support ctx <= 32 and head dim <= 32");
    this.pipes = {};
    const pipe = (label, code, constants) => {
      const shader = device.createShaderModule({ code, label });
      this.pipes[label] = device.createComputePipeline({ layout: "auto", compute: { module: shader, entryPoint: "main", constants }, label });
    };
    for (const [name, code] of Object.entries(K)) pipe(name, code);
    pipe("matmulSmall", K.matmul, { RT: 2 });
    this._allocActivations();
    this.plans = new Map();
    this.staging = new Map();
  }

  _buf(bytes, data, usage = STORAGE) {
    const size = Math.max(16, Math.ceil(bytes / 16) * 16);
    const buf = this.device.createBuffer({ size, usage, mappedAtCreation: !!data });
    if (data) {
      new data.constructor(buf.getMappedRange()).set(data);
      buf.unmap();
    }
    return buf;
  }

  _allocActivations() {
    const { vocab: V, d: C, layers: L, heads: H, ctx } = this.cfg;
    const R = this.maxRows;
    const f = (n) => this._buf(n * 4);
    const att = R * H * ctx; // B*H*T*T <= rows*H*ctx
    this.act = {
      tokens: f(R), targets: f(R), gradLogp: f(R), logp: f(R), lse: f(R),
      csr: f(1 + R + (R + 1) + R),
      x0: f(R * C),
      layers: Array.from({ length: L }, () => ({
        ln1: f(R * C), ln1m: f(R), ln1r: f(R), qkv: f(R * 3 * C), att: f(att), atty: f(R * C),
        xmid: f(R * C), ln2: f(R * C), ln2m: f(R), ln2r: f(R), fc: f(R * 4 * C), gelu: f(R * 4 * C), xout: f(R * C),
      })),
      lnf: f(R * C), lnfm: f(R), lnfr: f(R),
      logits: f(R * V),
      last: f(R * C), lastLogits: f(R * V),
      dres: f(R * C), dC: f(R * C), dqkv: f(R * 3 * C), dfc: f(R * 4 * C),
      partial: f(256), norm: f(4), scratch: f(SCRATCH_FLOATS),
      dummy: f(4),
    };
    this.adamUniform = this.device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  }

  // --- plan building ------------------------------------------------------------------------

  _uniform(values) {
    const buf = this.device.createBuffer({ size: Math.ceil((values.length * 4) / 16) * 16, usage: GPUBufferUsage.UNIFORM, mappedAtCreation: true });
    new Uint32Array(buf.getMappedRange()).set(values);
    buf.unmap();
    return buf;
  }

  /** A binding: a whole buffer, or a named parameter / its gradient slice. */
  _p(name) { const { off, n } = this.layout[name]; return { buffer: this.params, offset: off * 4, size: n * 4 }; }
  _g(name) { const { off, n } = this.layout[name]; return { buffer: this.grads, offset: off * 4, size: n * 4 }; }

  _op(ops, pipe, uniform, bindings, wg) {
    const entries = [];
    let b = 0;
    if (uniform) entries.push({ binding: b++, resource: { buffer: uniform instanceof GPUBuffer ? uniform : this._uniform(uniform) } });
    for (const r of bindings) entries.push({ binding: b++, resource: r instanceof GPUBuffer ? { buffer: r } : r });
    const bindGroup = this.device.createBindGroup({ layout: this.pipes[pipe].getBindGroupLayout(0), entries });
    ops.push({ pipe: this.pipes[pipe], bindGroup, wg: Array.isArray(wg) ? wg : [wg] });
  }

  // Small outputs use 32x32 tiles and split K so the GPU isn't left with a handful of workgroups.
  _matmul(ops, { A, B, C, M, N, K: Kd, transA = 0, transB = 0, accumulate = 0, bias = null, res = null }) {
    const a = this.act;
    const big = Math.ceil(M / 64) * Math.ceil(N / 64) >= TARGET_WORKGROUPS / 2;
    const tile = big ? 64 : 32;
    const [gx, gy] = [Math.ceil(N / tile), Math.ceil(M / tile)];
    let splits = Math.min(Math.ceil(TARGET_WORKGROUPS / (gx * gy)), Math.floor(Kd / 64), Math.floor(SCRATCH_FLOATS / (M * N)));
    splits = Math.max(1, splits);
    const kChunk = Math.ceil(Kd / splits / 16) * 16;
    splits = Math.ceil(Kd / kChunk);
    const u = [M, N, Kd, transA, transB, accumulate, bias ? 1 : 0, res ? 1 : 0, kChunk, splits];
    this._op(ops, big ? "matmul" : "matmulSmall", u, [A, B, C, bias ?? a.dummy, res ?? a.dummy, a.scratch], [gx, gy, splits]);
    if (splits > 1) this._op(ops, "matmulReduce", u, [a.scratch, C, bias ?? a.dummy, res ?? a.dummy], this._elementwise(M * N));
  }

  _elementwise(n) { return grid(Math.ceil(n / 256)); }

  _forwardOps(ops, B, T, { allLogits }) {
    const { vocab: V, d: C, layers: L, heads: H } = this.cfg;
    const M = B * T;
    const a = this.act;
    this._op(ops, "encoderFwd", [M, T, C], [a.tokens, this._p("wte.weight"), this._p("wpe.weight"), a.x0], this._elementwise(M * C));
    let x = a.x0;
    for (let l = 0; l < L; l++) {
      const s = a.layers[l];
      const p = `blocks.${l}.`;
      this._op(ops, "layernormFwd", [M, C], [x, this._p(p + "ln1.weight"), this._p(p + "ln1.bias"), s.ln1, s.ln1m, s.ln1r], grid(M));
      this._matmul(ops, { A: s.ln1, B: this._p(p + "qkv.weight"), C: s.qkv, M, N: 3 * C, K: C, transB: 1, bias: this._p(p + "qkv.bias") });
      this._op(ops, "attentionFwd", [B, T, C, H], [s.qkv, s.att, s.atty], B * H);
      this._matmul(ops, { A: s.atty, B: this._p(p + "proj.weight"), C: s.xmid, M, N: C, K: C, transB: 1, bias: this._p(p + "proj.bias"), res: x });
      this._op(ops, "layernormFwd", [M, C], [s.xmid, this._p(p + "ln2.weight"), this._p(p + "ln2.bias"), s.ln2, s.ln2m, s.ln2r], grid(M));
      this._matmul(ops, { A: s.ln2, B: this._p(p + "fc.weight"), C: s.fc, M, N: 4 * C, K: C, transB: 1, bias: this._p(p + "fc.bias") });
      this._op(ops, "geluFwd", [M * 4 * C], [s.fc, s.gelu], this._elementwise(M * 4 * C));
      this._matmul(ops, { A: s.gelu, B: this._p(p + "fc_proj.weight"), C: s.xout, M, N: C, K: 4 * C, transB: 1, bias: this._p(p + "fc_proj.bias"), res: s.xmid });
      x = s.xout;
    }
    this._op(ops, "layernormFwd", [M, C], [x, this._p("lnf.weight"), this._p("lnf.bias"), a.lnf, a.lnfm, a.lnfr], grid(M));
    if (allLogits) {
      this._matmul(ops, { A: a.lnf, B: this._p("wte.weight"), C: a.logits, M, N: V, K: C, transB: 1 });
      this._op(ops, "logprobFwd", [M, V], [a.logits, a.targets, a.logp, a.lse], grid(M));
    } else {
      this._op(ops, "gatherRows", [B, T, C, T - 1], [a.lnf, a.last], Math.ceil((B * C) / 64));
      this._matmul(ops, { A: a.last, B: this._p("wte.weight"), C: a.lastLogits, M: B, N: V, K: C, transB: 1 });
    }
  }

  _backwardOps(ops, B, T) {
    const { vocab: V, d: C, layers: L, heads: H } = this.cfg;
    const M = B * T;
    const a = this.act;
    const lnParamBwd = (dy, x, m, r, w, b) =>
      this._op(ops, "layernormParamBwd", [M, C], [dy, x, m, r, this._g(w), this._g(b)], Math.ceil(C / 64));
    const linearBwd = (dY, X, W, bias, dX, N, Kd) => {
      // Y[M,N] = X[M,K] W^T + b  =>  dX = dY W, dW += dY^T X, db += colsum(dY)
      this._matmul(ops, { A: dY, B: this._p(W), C: dX, M, N: Kd, K: N });
      this._matmul(ops, { A: dY, B: X, C: this._g(W), M: N, N: Kd, K: M, transA: 1, accumulate: 1 });
      this._op(ops, "colsum", [M, N], [dY, this._g(bias)], Math.ceil(N / 64));
    };

    this._op(ops, "logprobBwd", [M, V], [a.logits, a.targets, a.lse, a.gradLogp], grid(M));
    this._matmul(ops, { A: a.logits, B: this._p("wte.weight"), C: a.dC, M, N: C, K: V });
    this._matmul(ops, { A: a.logits, B: a.lnf, C: this._g("wte.weight"), M: V, N: C, K: M, transA: 1, accumulate: 1 });
    const xLast = a.layers[L - 1].xout;
    this._op(ops, "layernormBwd", [M, C, 0], [a.dC, xLast, this._p("lnf.weight"), a.lnfm, a.lnfr, a.dres], grid(M));
    lnParamBwd(a.dC, xLast, a.lnfm, a.lnfr, "lnf.weight", "lnf.bias");

    for (let l = L - 1; l >= 0; l--) {
      const s = a.layers[l];
      const p = `blocks.${l}.`;
      const xin = l === 0 ? a.x0 : a.layers[l - 1].xout;
      // MLP branch: xout = xmid + fc_proj(gelu(fc(ln2(xmid))))
      linearBwd(a.dres, s.gelu, p + "fc_proj.weight", p + "fc_proj.bias", a.dfc, C, 4 * C);
      this._op(ops, "geluBwd", [M * 4 * C], [s.fc, a.dfc], this._elementwise(M * 4 * C));
      linearBwd(a.dfc, s.ln2, p + "fc.weight", p + "fc.bias", a.dC, 4 * C, C);
      this._op(ops, "layernormBwd", [M, C, 1], [a.dC, s.xmid, this._p(p + "ln2.weight"), s.ln2m, s.ln2r, a.dres], grid(M));
      lnParamBwd(a.dC, s.xmid, s.ln2m, s.ln2r, p + "ln2.weight", p + "ln2.bias");
      // Attention branch: xmid = xin + proj(attn(qkv(ln1(xin))))
      linearBwd(a.dres, s.atty, p + "proj.weight", p + "proj.bias", a.dC, C, C);
      this._op(ops, "attentionBwd", [B, T, C, H], [s.qkv, s.att, a.dC, a.dqkv], B * H);
      linearBwd(a.dqkv, s.ln1, p + "qkv.weight", p + "qkv.bias", a.dC, 3 * C, C);
      this._op(ops, "layernormBwd", [M, C, 1], [a.dC, xin, this._p(p + "ln1.weight"), s.ln1m, s.ln1r, a.dres], grid(M));
      lnParamBwd(a.dC, xin, s.ln1m, s.ln1r, p + "ln1.weight", p + "ln1.bias");
    }
    this._op(ops, "encoderBwdWpe", [B, T, C], [a.dres, this._g("wpe.weight")], Math.ceil((T * C) / 64));
    this._op(ops, "encoderBwdWte", [C, this.maxRows], [a.dres, a.csr, this._g("wte.weight")], this._elementwise(Math.min(M, V) * C));
  }

  _stepOps(ops) {
    const a = this.act;
    this._op(ops, "sumsqPartial", [this.nParams], [this.grads, a.partial], 256);
    this._op(ops, "sumsqFinal", null, [a.partial, a.norm], 1);
    this._op(ops, "adamw", this.adamUniform, [this.params, this.grads, this.adamM, this.adamV, a.norm], this._elementwise(this.nParams / 4));
  }

  _plan(kind, B, T) {
    const key = `${kind}:${B}:${T}`;
    let plan = this.plans.get(key);
    if (plan) return plan;
    if (B * T > this.maxRows) throw new Error(`batch ${B}x${T} exceeds maxRows ${this.maxRows}`);
    if (T > this.cfg.ctx) throw new Error(`length ${T} exceeds ctx ${this.cfg.ctx}`);
    plan = [];
    if (kind === "gen") this._forwardOps(plan, B, T, { allLogits: false });
    if (kind === "fwd" || kind === "train") this._forwardOps(plan, B, T, { allLogits: true });
    if (kind === "bwd" || kind === "train") this._backwardOps(plan, B, T);
    if (kind === "bwd" || kind === "train") this._stepOps(plan);
    this.plans.set(key, plan);
    return plan;
  }

  _encode(enc, plan) {
    const pass = enc.beginComputePass();
    for (const { pipe, bindGroup, wg } of plan) {
      pass.setPipeline(pipe);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(...wg);
    }
    pass.end();
  }

  async _read(src, floats) {
    const bytes = floats * 4;
    let buf = this.staging.get(bytes);
    if (!buf) {
      buf = this.device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      this.staging.set(bytes, buf);
    }
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(src, 0, buf, 0, bytes);
    this.device.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(buf.getMappedRange().slice(0));
    buf.unmap();
    return out;
  }

  // --- inputs ---------------------------------------------------------------------------------

  _upload(tokens, B, T) {
    const M = B * T;
    if (tokens.length !== M) throw new Error(`expected ${M} tokens, got ${tokens.length}`);
    if (M > this.maxRows) throw new Error(`batch ${B}x${T} exceeds maxRows ${this.maxRows}`);
    const tok = Uint32Array.from(tokens);
    const tgt = new Int32Array(M);
    for (let m = 0; m < M; m++) tgt[m] = m % T === T - 1 ? -1 : tok[m + 1];
    const q = this.device.queue;
    q.writeBuffer(this.act.tokens, 0, tok);
    q.writeBuffer(this.act.targets, 0, tgt);

    // Group positions by token id for the embedding backward (see encoderBwdWte).
    const cap = this.maxRows;
    const byTok = new Map();
    for (let m = 0; m < M; m++) {
      const list = byTok.get(tok[m]);
      if (list) list.push(m);
      else byTok.set(tok[m], [m]);
    }
    const csr = new Uint32Array(1 + cap + cap + 1 + M);
    csr[0] = byTok.size;
    let u = 0, k = 0;
    const offs = 1 + cap, pos0 = offs + cap + 1;
    for (const [t, list] of byTok) {
      csr[1 + u] = t;
      csr[offs + u] = k;
      for (const m of list) csr[pos0 + k++] = m;
      u++;
    }
    csr[offs + u] = k;
    q.writeBuffer(this.act.csr, 0, csr);
    this._shape = [B, T];
  }

  _writeAdam({ lr, beta1 = 0.9, beta2 = 0.95, eps = 1e-8, weightDecay = 0.1, maxGradNorm = 1.0 }) {
    this.adamT++;
    const buf = new ArrayBuffer(48);
    const f = new Float32Array(buf), u = new Uint32Array(buf);
    f.set([lr, beta1, beta2, eps, weightDecay, 1 - beta1 ** this.adamT, 1 - beta2 ** this.adamT, maxGradNorm]);
    u[8] = this.nParams;
    u[9] = this.decayEnd;
    this.device.queue.writeBuffer(this.adamUniform, 0, buf);
  }

  // --- public API -------------------------------------------------------------------------------

  /**
   * Forward pass. Returns logp[b*T+t] = log p(tokens[b,t+1] | tokens[b,:t+1]) (0 at t = T-1).
   * Activations stay on the GPU so a following backward() reuses them.
   */
  async forward(tokens, B, T) {
    this._upload(tokens, B, T);
    const enc = this.device.createCommandEncoder();
    this._encode(enc, this._plan("fwd", B, T));
    this.device.queue.submit([enc.finish()]);
    return this._read(this.act.logp, B * T);
  }

  /**
   * Backprop an arbitrary objective through the last forward(), then take one AdamW step.
   * gradLogp[m] = dLoss/dlogp[m]; e.g. LM loss uses -mask/count, DPO uses ±β·σ(·)/count per row.
   */
  async backward(gradLogp, opt) {
    const [B, T] = this._shape;
    this.device.queue.writeBuffer(this.act.gradLogp, 0, Float32Array.from(gradLogp));
    this._writeAdam(opt);
    const enc = this.device.createCommandEncoder();
    this._encode(enc, this._plan("bwd", B, T));
    this.device.queue.submit([enc.finish()]);
    await this.device.queue.onSubmittedWorkDone();
  }

  /** Fused forward + backward + step for objectives whose gradLogp doesn't depend on the outputs. */
  async trainStep(tokens, B, T, gradLogp, opt, { readLogp = false } = {}) {
    this._upload(tokens, B, T);
    this.device.queue.writeBuffer(this.act.gradLogp, 0, Float32Array.from(gradLogp));
    this._writeAdam(opt);
    const enc = this.device.createCommandEncoder();
    this._encode(enc, this._plan("train", B, T));
    this.device.queue.submit([enc.finish()]);
    // logp is computed before the update, so it reflects the pre-step weights.
    if (readLogp) return this._read(this.act.logp, B * T);
    await this.device.queue.onSubmittedWorkDone();
  }

  /** Next-token logits at the last position of each row. tokens: [B*T]. */
  async nextLogits(tokens, B, T) {
    this._upload(tokens, B, T);
    const enc = this.device.createCommandEncoder();
    this._encode(enc, this._plan("gen", B, T));
    this.device.queue.submit([enc.finish()]);
    return this._read(this.act.lastLogits, B * this.cfg.vocab);
  }

  /**
   * Sample n headlines from <|eot|>. Returns token arrays without the leading/trailing eot.
   * @param {{n:number, temperature?:number, topK?:number, prefix?:number[], rng?:() => number}} o
   */
  async generate({ n, temperature = 1.0, topK = 0, prefix = [], rng = Math.random }) {
    const V = this.cfg.vocab;
    const rows = Array.from({ length: n }, () => [0, ...prefix]);
    const done = new Array(n).fill(false);
    for (let T = rows[0].length; T < this.cfg.ctx && !done.every(Boolean); T++) {
      const logits = await this.nextLogits(rows.flat(), n, T);
      for (let b = 0; b < n; b++) {
        if (done[b]) { rows[b].push(0); continue; }
        const next = sampleLogits(logits.subarray(b * V, (b + 1) * V), temperature, topK, rng);
        rows[b].push(next);
        if (next === 0) done[b] = true;
      }
    }
    return rows.map((r) => { const body = r.slice(1); const end = body.indexOf(0); return end < 0 ? body : body.slice(0, end); });
  }

  /** Copy current weights back to the CPU (e.g. to snapshot a frozen reference model). */
  async readWeights() {
    const flat = await this._read(this.params, this.nParams);
    return Object.fromEntries(Object.entries(this.layout).map(([k, { off, n }]) => [k, flat.slice(off, off + n)]));
  }

  async readGrads() {
    const flat = await this._read(this.grads, this.nParams);
    return Object.fromEntries(Object.entries(this.layout).map(([k, { off, n }]) => [k, flat.slice(off, off + n)]));
  }

  /** Overwrite weights and reset optimizer state (e.g. "reset to pretrained"). */
  setWeights(tensors) {
    const flat = new Float32Array(this.nParams);
    for (const [name, { off }] of Object.entries(this.layout)) flat.set(tensors[name], off);
    const q = this.device.queue;
    q.writeBuffer(this.params, 0, flat);
    const zeros = new Float32Array(this.nParams);
    q.writeBuffer(this.grads, 0, zeros);
    q.writeBuffer(this.adamM, 0, zeros);
    q.writeBuffer(this.adamV, 0, zeros);
    this.adamT = 0;
  }
}

export function sampleLogits(logits, temperature, topK, rng) {
  const V = logits.length;
  let mx = -Infinity, arg = 0;
  for (let i = 0; i < V; i++) if (logits[i] > mx) { mx = logits[i]; arg = i; }
  if (temperature <= 0) return arg;
  // Keep logits >= the k-th largest (a native numeric sort of a copy is fast at V = 4096).
  const cut = topK > 0 && topK < V ? Float32Array.from(logits).sort()[V - topK] : -Infinity;
  let total = 0;
  const w = new Float32Array(V);
  for (let i = 0; i < V; i++) if (logits[i] >= cut) total += (w[i] = Math.exp((logits[i] - mx) / temperature));
  let r = rng() * total;
  for (let i = 0; i < V; i++) if (w[i] > 0 && (r -= w[i]) <= 0) return i;
  return arg;
}
