// Simulated A/B clicks: a keyword "clickbait judge" picks between two fresh samples.
// Shows how fast DPO moves the policy and whether it stays fluent.
// Usage: node test/simulate.mjs public/headline-gpt.bin [clicks] [lr] [beta] [steps] [replay]
import { readFileSync } from "node:fs";
import { getDevice } from "./gpu.mjs";
import { DPOSession } from "../src/rlhf.js";

const [file, clicks = 40, lr = 3e-4, beta = 0.1, steps = 2, replay = 7, sft = 1] = process.argv.slice(2).map((x, i) => (i ? +x : x));
// A consistent preference the base model can already express sometimes: crime and accidents.
const JUDGE = process.env.JUDGE ?? "crime";
const LEXICON = {
  crime: /\b(police|charged|court|murder|dies|died|crash|killed|arrested|death|stabbing|assault|jail|shot|dead)\b/g,
  bait: /\b(you|your|this|these|why|how|what|shocking|secret|amazing|best|worst|never|believe|will|reasons|things|ways|here|everyone|need|know)\b|[?!]|\b\d+\b/g,
}[JUDGE];
// Like a person, the judge dislikes repetition: each repeated word costs as much as a hit.
const score = (s) => {
  const words = s.split(" ");
  return (s.match(LEXICON) ?? []).length - (words.length - new Set(words).size);
};
const device = await getDevice();
const session = DPOSession.fromBytes(device, readFileSync(file).buffer, { lr, beta, stepsPerClick: steps, replay, sft });
// bait = share of samples the judge likes; refNLL = per-token NLL under the frozen pretrained model
// (rises when text turns to gibberish); distinct = share of 48 samples that are unique.
const probe = async () => {
  const rows = [];
  for (let k = 0; k < 3; k++) rows.push(...(await session.policy.generate({ n: 16, temperature: session.temperature, topK: session.topK })));
  const xs = rows.map((ids) => session.text(ids));
  const kept = rows.filter((r) => r.length);
  const ref = await session._refLogp(kept);
  const tokens = kept.reduce((a, r) => a + r.length + 1, 0);
  return {
    bait: xs.filter((s) => score(s) > 0).length / xs.length,
    refNLL: -ref.reduce((a, b) => a + b, 0) / tokens,
    distinct: new Set(xs).size / xs.length,
    examples: xs.slice(0, 3),
  };
};
const fmt = (p) => `liked ${p.bait.toFixed(2)}  refNLL ${p.refNLL.toFixed(2)}  distinct ${p.distinct.toFixed(2)}`;
const before = await probe();
console.log(`lr ${lr} beta ${beta} steps ${steps} replay ${replay} sft ${sft}`);
console.log(`before: ${fmt(before)}`, before.examples);
let ms = 0;
for (let c = 1; c <= clicks; c++) {
  const [a, b] = await session.samplePair();
  const [sa, sb] = [score(session.text(a)), score(session.text(b))];
  const aWins = sa === sb ? Math.random() < 0.5 : sa > sb;
  const stats = await session.prefer(aWins ? a : b, aWins ? b : a);
  ms += stats.ms;
  if (c % 20 === 0) {
    const p = await probe();
    console.log(`click ${c}: loss ${stats.loss.toFixed(3)} acc ${stats.accuracy.toFixed(2)}  ${fmt(p)}  ${(ms / c).toFixed(1)} ms/click`, p.examples);
  }
}
process.exit(0);
