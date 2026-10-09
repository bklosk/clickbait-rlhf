// Timing for the shapes an in-browser RLHF loop uses. Usage: node test/bench.mjs model.bin
import { readFileSync } from "node:fs";
import { getDevice } from "./gpu.mjs";
import { GPT } from "../src/gpt.js";
import { parseModel } from "../src/format.js";
import { Tokenizer } from "../src/tokenizer.js";

const { config, tensors, tokenizer } = parseModel(readFileSync(process.argv[2]).buffer);
const device = await getDevice();
const gpt = new GPT(device, config, tensors, { maxRows: 2048 });
const tok = new Tokenizer(tokenizer);
const opt = { lr: 1e-4 };

async function time(label, fn, reps = 20) {
  await fn(); await fn(); // warm up (pipeline + plan creation)
  const t0 = performance.now();
  for (let i = 0; i < reps; i++) await fn();
  const ms = (performance.now() - t0) / reps;
  console.log(`${label.padEnd(44)} ${ms.toFixed(2)} ms`);
  return ms;
}

for (const [B, T] of [[2, 24], [8, 24], [16, 32], [64, 32]]) {
  const tokens = Array.from({ length: B * T }, (_, i) => (i * 7919) % config.vocab);
  const g = new Float32Array(B * T).fill(-1 / (B * T));
  const ms = await time(`train step  B=${B} T=${T} (fwd+bwd+AdamW)`, () => gpt.trainStep(tokens, B, T, g, opt));
  console.log(`${"".padEnd(44)} ${((B * T) / ms * 1000).toFixed(0)} tokens/s`);
  await time(`fwd -> readback -> bwd  B=${B} T=${T} (DPO-style)`, async () => { await gpt.forward(tokens, B, T); await gpt.backward(g, opt); });
}
await time("generate 8 headlines (temp 1, top-k 50)", () => gpt.generate({ n: 8, topK: 50 }), 5);
await time("generate 32 headlines (temp 1, top-k 50)", () => gpt.generate({ n: 32, topK: 50 }), 5);

gpt.setWeights(tensors);
for (const ids of await gpt.generate({ n: 8, temperature: 0.9, topK: 50 })) console.log("  ", tok.decode(ids));
process.exit(0);
