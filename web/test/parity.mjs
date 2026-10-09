// Compares the WebGPU forward/backward/AdamW against PyTorch (float64) numbers from
// `pretrain/export.py --fixture`. Usage: node test/parity.mjs ../data/fixture.bin
import { readFileSync } from "node:fs";
import { getDevice } from "./gpu.mjs";
import { GPT } from "../src/gpt.js";
import { parseModel } from "../src/format.js";

const { config, tensors: fx, meta } = parseModel(readFileSync(process.argv[2]).buffer);
const { B, T, tokens, opt } = meta;
const strip = (prefix) => Object.fromEntries(Object.entries(fx).filter(([k]) => k.startsWith(prefix)).map(([k, v]) => [k.slice(prefix.length), v]));
const weights = strip("w/"), grads = strip("g/"), after = strip("after/");

function relErr(got, want) {
  let num = 0, den = 0, maxAbs = 0;
  for (let i = 0; i < want.length; i++) {
    const d = got[i] - want[i];
    num += d * d; den += want[i] * want[i];
    maxAbs = Math.max(maxAbs, Math.abs(d));
  }
  return { rel: Math.sqrt(num / Math.max(den, 1e-30)), maxAbs };
}

const device = await getDevice();
const gpt = new GPT(device, config, weights, { maxRows: 512 });
let failed = 0;
const check = (name, got, want, tol) => {
  const { rel, maxAbs } = relErr(got, want);
  const ok = rel < tol && Number.isFinite(rel);
  if (!ok) failed++;
  if (!ok || process.env.VERBOSE) console.log(`${ok ? "ok  " : "FAIL"} ${name.padEnd(28)} rel ${rel.toExponential(2)}  maxAbs ${maxAbs.toExponential(2)}`);
  return rel;
};

const logp = await gpt.forward(tokens, B, T);
check("logp", logp, fx.logp, 1e-4);

// AdamW zeroes the grads it consumes, so run the backward ops alone first and read them.
const plan = gpt._plan("bwd", B, T).slice(0, -3); // drop sumsq/sumsq/adamw
device.queue.writeBuffer(gpt.act.gradLogp, 0, fx.grad_logp);
let enc = device.createCommandEncoder();
gpt._encode(enc, plan);
device.queue.submit([enc.finish()]);
const g = await gpt.readGrads();
let worst = 0;
for (const k of Object.keys(grads)) worst = Math.max(worst, check(`grad ${k}`, g[k], grads[k], 2e-3));
console.log(`grads: worst relative error ${worst.toExponential(2)} over ${Object.keys(grads).length} tensors`);

// Now the optimizer step on those same grads.
gpt._writeAdam(opt);
enc = device.createCommandEncoder();
gpt._encode(enc, gpt._plan("bwd", B, T).slice(-3));
device.queue.submit([enc.finish()]);
const w = await gpt.readWeights();
let worstStep = 0;
for (const k of Object.keys(after)) {
  // Compare the update (after - before), which is far more sensitive than the weights themselves.
  const du = w[k].map((x, i) => x - weights[k][i]);
  const dw = after[k].map((x, i) => x - weights[k][i]);
  worstStep = Math.max(worstStep, check(`step ${k}`, du, dw, 2e-2));
}
console.log(`adamw: worst relative error of the update ${worstStep.toExponential(2)}`);

// Fused path must agree with the split path's pre-step logp.
gpt.setWeights(weights);
const fused = await gpt.trainStep(tokens, B, T, fx.grad_logp, opt, { readLogp: true });
check("fused trainStep logp", fused, fx.logp, 1e-4);

console.log(failed ? `${failed} checks FAILED` : "all parity checks passed");
process.exit(failed ? 1 : 0);
