// WGSL compute kernels for the headline GPT. Each forward kernel has a hand-written
// backward twin; pretrain/model.py is the reference they are tested against.

// Linear index for 1D elementwise kernels dispatched as a 2D grid of 256-wide workgroups.
const IDX = /* wgsl */ `
fn flat(gid: vec3u, nwg: vec3u) -> u32 { return gid.x + gid.y * nwg.x * 256u; }
fn row_of(wg: vec3u, nwg: vec3u) -> u32 { return wg.x + wg.y * nwg.x; }
`;

// C[M,N] (=|+=) A[M,K]·B[K,N] (+ bias[N]) (+ R[M,N]).
// transA: A is stored [K,M]. transB: B is stored [N,K] (i.e. torch Linear weight layout).
// Each 16x16 workgroup computes a (16·RT)² tile. With splits > 1 (split-K, for small M·N),
// workgroup z covers K range [z·kChunk, (z+1)·kChunk) and writes raw partials to scratch;
// matmulReduce then sums them and applies the epilogue.
const MATMUL_COMMON = /* wgsl */ `
struct P { M: u32, N: u32, K: u32, transA: u32, transB: u32, accumulate: u32, hasBias: u32, hasRes: u32, kChunk: u32, splits: u32 }
@group(0) @binding(0) var<uniform> p: P;
`;

const MATMUL_EPILOGUE = /* wgsl */ `
fn epilogue(m: u32, n: u32, acc: f32) {
  var v = acc;
  if (p.hasBias == 1u) { v += bias[n]; }
  if (p.hasRes == 1u) { v += R[m * p.N + n]; }
  if (p.accumulate == 1u) { v += C[m * p.N + n]; }
  C[m * p.N + n] = v;
}
`;

export const matmul = /* wgsl */ `
${MATMUL_COMMON}
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;
@group(0) @binding(4) var<storage, read> bias: array<f32>;
@group(0) @binding(5) var<storage, read> R: array<f32>;
@group(0) @binding(6) var<storage, read_write> scratch: array<f32>;
${MATMUL_EPILOGUE}
override RT: u32 = 4u;

var<workgroup> As: array<f32, 1024>; // [16 k][TILE m]
var<workgroup> Bs: array<f32, 1024>; // [16 k][TILE n]

fn loadA(m: u32, k: u32, kEnd: u32) -> f32 {
  if (m >= p.M || k >= kEnd) { return 0.0; }
  if (p.transA == 1u) { return A[k * p.M + m]; }
  return A[m * p.K + k];
}
fn loadB(k: u32, n: u32, kEnd: u32) -> f32 {
  if (k >= kEnd || n >= p.N) { return 0.0; }
  if (p.transB == 1u) { return B[n * p.K + k]; }
  return B[k * p.N + n];
}

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let TILE = 16u * RT;
  let tid = lid.y * 16u + lid.x;
  let m0 = wg.y * TILE;
  let n0 = wg.x * TILE;
  let kBeg = wg.z * p.kChunk;
  let kEnd = min(p.K, kBeg + p.kChunk);
  var acc: array<f32, 16>;
  for (var k0 = kBeg; k0 < kEnd; k0 += 16u) {
    for (var i = 0u; i < RT; i++) {
      let idx = tid + i * 256u;
      // Pick the tile walk so consecutive threads read consecutive addresses.
      var am = idx / 16u; var ak = idx % 16u;
      if (p.transA == 1u) { am = idx % TILE; ak = idx / TILE; }
      As[ak * TILE + am] = loadA(m0 + am, k0 + ak, kEnd);
      var bk = idx / TILE; var bn = idx % TILE;
      if (p.transB == 1u) { bk = idx % 16u; bn = idx / 16u; }
      Bs[bk * TILE + bn] = loadB(k0 + bk, n0 + bn, kEnd);
    }
    workgroupBarrier();
    for (var kk = 0u; kk < 16u; kk++) {
      var a: array<f32, 4>;
      var b: array<f32, 4>;
      for (var i = 0u; i < RT; i++) {
        a[i] = As[kk * TILE + lid.y * RT + i];
        b[i] = Bs[kk * TILE + lid.x * RT + i];
      }
      for (var i = 0u; i < RT; i++) {
        for (var j = 0u; j < RT; j++) { acc[i * 4u + j] += a[i] * b[j]; }
      }
    }
    workgroupBarrier();
  }
  for (var i = 0u; i < RT; i++) {
    let m = m0 + lid.y * RT + i;
    if (m >= p.M) { continue; }
    for (var j = 0u; j < RT; j++) {
      let n = n0 + lid.x * RT + j;
      if (n >= p.N) { continue; }
      if (p.splits > 1u) { scratch[(wg.z * p.M + m) * p.N + n] = acc[i * 4u + j]; }
      else { epilogue(m, n, acc[i * 4u + j]); }
    }
  }
}
`;

export const matmulReduce = /* wgsl */ `
${MATMUL_COMMON}
@group(0) @binding(1) var<storage, read> scratch: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read> R: array<f32>;
${MATMUL_EPILOGUE}
${IDX}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
  let i = flat(gid, nwg);
  if (i >= p.M * p.N) { return; }
  var s = 0.0;
  for (var z = 0u; z < p.splits; z++) { s += scratch[z * p.M * p.N + i]; }
  epilogue(i / p.N, i % p.N, s);
}
`;

export const encoderFwd = /* wgsl */ `
struct P { M: u32, T: u32, C: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> tok: array<u32>;
@group(0) @binding(2) var<storage, read> wte: array<f32>;
@group(0) @binding(3) var<storage, read> wpe: array<f32>;
@group(0) @binding(4) var<storage, read_write> x: array<f32>;
${IDX}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
  let i = flat(gid, nwg);
  if (i >= p.M * p.C) { return; }
  let m = i / p.C;
  let c = i % p.C;
  x[i] = wte[tok[m] * p.C + c] + wpe[(m % p.T) * p.C + c];
}
`;

// One 64-thread workgroup per row.
const ROW_REDUCE = /* wgsl */ `
var<workgroup> red: array<f32, 64>;
fn reduce_sum(lid: u32, v: f32) -> f32 {
  red[lid] = v;
  workgroupBarrier();
  for (var s = 32u; s > 0u; s >>= 1u) {
    if (lid < s) { red[lid] += red[lid + s]; }
    workgroupBarrier();
  }
  let out = red[0];
  workgroupBarrier();
  return out;
}
`;

export const layernormFwd = /* wgsl */ `
struct P { M: u32, C: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
@group(0) @binding(5) var<storage, read_write> mean: array<f32>;
@group(0) @binding(6) var<storage, read_write> rstd: array<f32>;
${IDX}
${ROW_REDUCE}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(num_workgroups) nwg: vec3u, @builtin(local_invocation_index) lid: u32) {
  let row = row_of(wg, nwg);
  let live = row < p.M;
  let base = row * p.C;
  var s = 0.0;
  if (live) { for (var c = lid; c < p.C; c += 64u) { s += x[base + c]; } }
  let mu = reduce_sum(lid, s) / f32(p.C);
  var s2 = 0.0;
  if (live) { for (var c = lid; c < p.C; c += 64u) { let d = x[base + c] - mu; s2 += d * d; } }
  let r = inverseSqrt(reduce_sum(lid, s2) / f32(p.C) + 1e-5);
  if (!live) { return; }
  for (var c = lid; c < p.C; c += 64u) { y[base + c] = (x[base + c] - mu) * r * w[c] + b[c]; }
  if (lid == 0u) { mean[row] = mu; rstd[row] = r; }
}
`;

// dx (+)= LayerNorm backward w.r.t. its input.
export const layernormBwd = /* wgsl */ `
struct P { M: u32, C: u32, accumulate: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> dy: array<f32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read> w: array<f32>;
@group(0) @binding(4) var<storage, read> mean: array<f32>;
@group(0) @binding(5) var<storage, read> rstd: array<f32>;
@group(0) @binding(6) var<storage, read_write> dx: array<f32>;
${IDX}
${ROW_REDUCE}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(num_workgroups) nwg: vec3u, @builtin(local_invocation_index) lid: u32) {
  let row = row_of(wg, nwg);
  let live = row < p.M;
  let base = row * p.C;
  var mu = 0.0; var r = 0.0;
  if (live) { mu = mean[row]; r = rstd[row]; }
  var s1 = 0.0; var s2 = 0.0;
  if (live) {
    for (var c = lid; c < p.C; c += 64u) {
      let dn = dy[base + c] * w[c];
      s1 += dn;
      s2 += dn * (x[base + c] - mu) * r;
    }
  }
  let m1 = reduce_sum(lid, s1) / f32(p.C);
  let m2 = reduce_sum(lid, s2) / f32(p.C);
  if (!live) { return; }
  for (var c = lid; c < p.C; c += 64u) {
    let xhat = (x[base + c] - mu) * r;
    var v = r * (dy[base + c] * w[c] - m1 - xhat * m2);
    if (p.accumulate == 1u) { v += dx[base + c]; }
    dx[base + c] = v;
  }
}
`;

// dw[c] += Σ_m dy·x̂, db[c] += Σ_m dy. One thread per channel.
export const layernormParamBwd = /* wgsl */ `
struct P { M: u32, C: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> dy: array<f32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read> mean: array<f32>;
@group(0) @binding(4) var<storage, read> rstd: array<f32>;
@group(0) @binding(5) var<storage, read_write> dw: array<f32>;
@group(0) @binding(6) var<storage, read_write> db: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let c = gid.x;
  if (c >= p.C) { return; }
  var sw = 0.0; var sb = 0.0;
  for (var m = 0u; m < p.M; m++) {
    let g = dy[m * p.C + c];
    sw += g * (x[m * p.C + c] - mean[m]) * rstd[m];
    sb += g;
  }
  dw[c] += sw;
  db[c] += sb;
}
`;

// db[n] += Σ_m dY[m,n]
export const colsum = /* wgsl */ `
struct P { M: u32, N: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> dY: array<f32>;
@group(0) @binding(2) var<storage, read_write> db: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let n = gid.x;
  if (n >= p.N) { return; }
  var s = 0.0;
  for (var m = 0u; m < p.M; m++) { s += dY[m * p.N + n]; }
  db[n] += s;
}
`;

// Attention: one workgroup per (b, h) with one thread per position; the head's k and v (plus dy
// and dS in backward) live in workgroup memory, sized to fit WebGPU's default 16 KB.
// Limits: T <= 32, head dim <= 32 (checked in gpt.js).
// qkv rows are [q | k | v], each C wide, head h at columns h*hd.
const ATTN_COMMON = /* wgsl */ `
struct P { B: u32, T: u32, C: u32, H: u32 }
@group(0) @binding(0) var<uniform> p: P;
const MT = 32u;
var<workgroup> Ks: array<f32, 1024>; // [t][d]
var<workgroup> Vs: array<f32, 1024>;
fn load_kv(b: u32, h: u32, hd: u32, lid: u32) {
  let C3 = 3u * p.C;
  for (var e = lid; e < p.T * hd; e += MT) {
    let t = e / hd;
    let d = e % hd;
    let row = (b * p.T + t) * C3 + h * hd + d;
    Ks[t * hd + d] = qkv[row + p.C];
    Vs[t * hd + d] = qkv[row + 2u * p.C];
  }
}
`;

export const attentionFwd = /* wgsl */ `
${ATTN_COMMON}
@group(0) @binding(1) var<storage, read> qkv: array<f32>;
@group(0) @binding(2) var<storage, read_write> att: array<f32>; // [B,H,T,T] probabilities
@group(0) @binding(3) var<storage, read_write> y: array<f32>;   // [B*T, C]
@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) i: u32) {
  let b = wg.x / p.H;
  let h = wg.x % p.H;
  let hd = p.C / p.H;
  load_kv(b, h, hd, i);
  workgroupBarrier();
  if (i >= p.T) { return; }
  let scale = inverseSqrt(f32(hd));
  var q: array<f32, 32>;
  for (var d = 0u; d < hd; d++) { q[d] = qkv[(b * p.T + i) * 3u * p.C + h * hd + d]; }
  var s: array<f32, 32>;
  var mx = -1e30;
  for (var j = 0u; j <= i; j++) {
    var dot = 0.0;
    for (var d = 0u; d < hd; d++) { dot += q[d] * Ks[j * hd + d]; }
    s[j] = dot * scale;
    mx = max(mx, s[j]);
  }
  var sum = 0.0;
  for (var j = 0u; j <= i; j++) { s[j] = exp(s[j] - mx); sum += s[j]; }
  let arow = (wg.x * p.T + i) * p.T;
  for (var j = 0u; j < p.T; j++) {
    s[j] = select(0.0, s[j] / sum, j <= i);
    att[arow + j] = s[j];
  }
  let yo = (b * p.T + i) * p.C + h * hd;
  for (var d = 0u; d < hd; d++) {
    var acc = 0.0;
    for (var j = 0u; j <= i; j++) { acc += s[j] * Vs[j * hd + d]; }
    y[yo + d] = acc;
  }
}
`;

// dS[i,j] = P[i,j](dP[i,j] - Σ_j P dP) with dP = dy_i·v_j; then
// dq_i = scale Σ_j dS[i,j] k_j,  dk_j = scale Σ_i dS[i,j] q_i,  dv_j = Σ_i P[i,j] dy_i.
export const attentionBwd = /* wgsl */ `
${ATTN_COMMON}
@group(0) @binding(1) var<storage, read> qkv: array<f32>;
@group(0) @binding(2) var<storage, read> att: array<f32>;
@group(0) @binding(3) var<storage, read> dy: array<f32>;
@group(0) @binding(4) var<storage, read_write> dqkv: array<f32>;
var<workgroup> dYs: array<f32, 1024>;
var<workgroup> dSs: array<f32, 1024>; // [i][j]
@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) lid: u32) {
  let b = wg.x / p.H;
  let h = wg.x % p.H;
  let hd = p.C / p.H;
  let T = p.T;
  load_kv(b, h, hd, lid);
  for (var e = lid; e < T * hd; e += MT) {
    dYs[e] = dy[(b * T + e / hd) * p.C + h * hd + e % hd];
  }
  workgroupBarrier();
  let P0 = wg.x * T * T;
  let scale = inverseSqrt(f32(hd));
  let C3 = 3u * p.C;
  if (lid < T) {
    let i = lid;
    var dot = 0.0;
    for (var j = 0u; j <= i; j++) {
      var dp = 0.0;
      for (var d = 0u; d < hd; d++) { dp += dYs[i * hd + d] * Vs[j * hd + d]; }
      dSs[i * T + j] = dp;
      dot += dp * att[P0 + i * T + j];
    }
    for (var j = 0u; j < T; j++) { dSs[i * T + j] = select(0.0, att[P0 + i * T + j] * (dSs[i * T + j] - dot), j <= i); }
  }
  workgroupBarrier();
  if (lid >= T) { return; }
  let r = lid; // row i for dq, column j for dk/dv
  let base = (b * T + r) * C3 + h * hd;
  for (var d = 0u; d < hd; d++) {
    var dq = 0.0;
    for (var j = 0u; j <= r; j++) { dq += dSs[r * T + j] * Ks[j * hd + d]; }
    var dk = 0.0;
    var dv = 0.0;
    for (var i = r; i < T; i++) {
      dk += dSs[i * T + r] * qkv[(b * T + i) * C3 + h * hd + d];
      dv += att[P0 + i * T + r] * dYs[i * hd + d];
    }
    dqkv[base + d] = dq * scale;
    dqkv[base + p.C + d] = dk * scale;
    dqkv[base + 2u * p.C + d] = dv;
  }
}
`;

const GELU = /* wgsl */ `
const K0 = 0.7978845608028654; // sqrt(2/pi)
const K1 = 0.044715;
fn tanh_safe(u: f32) -> f32 { return tanh(clamp(u, -15.0, 15.0)); }
`;

export const geluFwd = /* wgsl */ `
struct P { n: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
${IDX}
${GELU}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
  let i = flat(gid, nwg);
  if (i >= p.n) { return; }
  let v = x[i];
  y[i] = 0.5 * v * (1.0 + tanh_safe(K0 * (v + K1 * v * v * v)));
}
`;

// In place: d <- d * gelu'(x)
export const geluBwd = /* wgsl */ `
struct P { n: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> d: array<f32>;
${IDX}
${GELU}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
  let i = flat(gid, nwg);
  if (i >= p.n) { return; }
  let v = x[i];
  let t = tanh_safe(K0 * (v + K1 * v * v * v));
  d[i] *= 0.5 * (1.0 + t) + 0.5 * v * (1.0 - t * t) * K0 * (1.0 + 3.0 * K1 * v * v);
}
`;

// One 256-thread workgroup per row: lse = logsumexp(logits), logp = logits[tgt] - lse.
const ROW_REDUCE_256 = /* wgsl */ `
var<workgroup> red: array<f32, 256>;
fn reduce256(lid: u32, v: f32, is_max: bool) -> f32 {
  red[lid] = v;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (lid < s) { red[lid] = select(red[lid] + red[lid + s], max(red[lid], red[lid + s]), is_max); }
    workgroupBarrier();
  }
  let out = red[0];
  workgroupBarrier();
  return out;
}
`;

export const logprobFwd = /* wgsl */ `
struct P { M: u32, V: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> logits: array<f32>;
@group(0) @binding(2) var<storage, read> tgt: array<i32>;
@group(0) @binding(3) var<storage, read_write> logp: array<f32>;
@group(0) @binding(4) var<storage, read_write> lse: array<f32>;
${IDX}
${ROW_REDUCE_256}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(num_workgroups) nwg: vec3u, @builtin(local_invocation_index) lid: u32) {
  let row = row_of(wg, nwg);
  let live = row < p.M;
  let base = row * p.V;
  var mx = -1e30;
  if (live) { for (var v = lid; v < p.V; v += 256u) { mx = max(mx, logits[base + v]); } }
  mx = reduce256(lid, mx, true);
  var s = 0.0;
  if (live) { for (var v = lid; v < p.V; v += 256u) { s += exp(logits[base + v] - mx); } }
  let l = mx + log(reduce256(lid, s, false));
  if (!live || lid != 0u) { return; }
  lse[row] = l;
  let t = tgt[row];
  logp[row] = select(0.0, logits[base + u32(max(t, 0))] - l, t >= 0);
}
`;

// In place: logits <- g[m] * (onehot(tgt) - softmax(logits))
export const logprobBwd = /* wgsl */ `
struct P { M: u32, V: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> logits: array<f32>;
@group(0) @binding(2) var<storage, read> tgt: array<i32>;
@group(0) @binding(3) var<storage, read> lse: array<f32>;
@group(0) @binding(4) var<storage, read> g: array<f32>;
${IDX}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(num_workgroups) nwg: vec3u, @builtin(local_invocation_index) lid: u32) {
  let row = row_of(wg, nwg);
  if (row >= p.M) { return; }
  let base = row * p.V;
  let gm = select(0.0, g[row], tgt[row] >= 0);
  let l = lse[row];
  let t = tgt[row];
  for (var v = lid; v < p.V; v += 256u) {
    let pr = exp(logits[base + v] - l);
    logits[base + v] = gm * (select(0.0, 1.0, i32(v) == t) - pr);
  }
}
`;

// dwpe[t,c] += Σ_b dx[b,t,c]
export const encoderBwdWpe = /* wgsl */ `
struct P { B: u32, T: u32, C: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> dx: array<f32>;
@group(0) @binding(2) var<storage, read_write> dwpe: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= p.T * p.C) { return; }
  var s = 0.0;
  for (var b = 0u; b < p.B; b++) { s += dx[b * p.T * p.C + i]; }
  dwpe[i] += s;
}
`;

// dwte[tok,c] += Σ dx[m,c] over positions m holding tok. The CPU groups positions by token
// (csr = [U, tokens[cap], offsets[cap+1], positions...]) so no float atomics are needed.
export const encoderBwdWte = /* wgsl */ `
struct P { C: u32, cap: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> dx: array<f32>;
@group(0) @binding(2) var<storage, read> csr: array<u32>;
@group(0) @binding(3) var<storage, read_write> dwte: array<f32>;
${IDX}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
  let i = flat(gid, nwg);
  let u = i / p.C;
  let c = i % p.C;
  if (u >= csr[0]) { return; }
  let tok = csr[1u + u];
  let offs = 1u + p.cap;
  let pos0 = offs + p.cap + 1u;
  var s = 0.0;
  for (var k = csr[offs + u]; k < csr[offs + u + 1u]; k++) { s += dx[csr[pos0 + k] * p.C + c]; }
  dwte[tok * p.C + c] += s;
}
`;

// out[b,:] = x[b*T + pos, :]
export const gatherRows = /* wgsl */ `
struct P { B: u32, T: u32, C: u32, pos: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= p.B * p.C) { return; }
  let b = i / p.C;
  out[i] = x[(b * p.T + p.pos) * p.C + i % p.C];
}
`;

// Global grad norm in two passes: 256 partial sums, then one workgroup finishes.
export const sumsqPartial = /* wgsl */ `
struct P { n: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> g: array<f32>;
@group(0) @binding(2) var<storage, read_write> partial: array<f32>;
${ROW_REDUCE_256}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) lid: u32) {
  var s = 0.0;
  for (var i = wg.x * 256u + lid; i < p.n; i += 65536u) { s += g[i] * g[i]; }
  let tot = reduce256(lid, s, false);
  if (lid == 0u) { partial[wg.x] = tot; }
}
`;

export const sumsqFinal = /* wgsl */ `
@group(0) @binding(0) var<storage, read> partial: array<f32>;
@group(0) @binding(1) var<storage, read_write> norm: array<f32>;
${ROW_REDUCE_256}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) lid: u32) {
  let tot = reduce256(lid, partial[lid], false);
  if (lid == 0u) { norm[0] = sqrt(tot); }
}
`;

// AdamW with global-norm clipping, 4 floats per thread; zeroes the gradient after use.
// n and decayEnd are multiples of 64 (gpt.js pads every tensor).
export const adamw = /* wgsl */ `
struct P { lr: f32, b1: f32, b2: f32, eps: f32, wd: f32, bc1: f32, bc2: f32, maxNorm: f32, n: u32, decayEnd: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> w: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> g: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> m: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> v: array<vec4f>;
@group(0) @binding(5) var<storage, read> norm: array<f32>;
${IDX}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
  let i = flat(gid, nwg);
  if (i * 4u >= p.n) { return; }
  var clip = 1.0;
  if (p.maxNorm > 0.0) { clip = min(1.0, p.maxNorm / (norm[0] + 1e-6)); }
  let gi = g[i] * clip;
  let mi = p.b1 * m[i] + (1.0 - p.b1) * gi;
  let vi = p.b2 * v[i] + (1.0 - p.b2) * gi * gi;
  m[i] = mi;
  v[i] = vi;
  let wd = select(0.0, p.wd, i * 4u < p.decayEnd);
  w[i] -= p.lr * ((mi / p.bc1) / (sqrt(vi / p.bc2) + p.eps) + wd * w[i]);
  g[i] = vec4f(0.0);
}
`;
