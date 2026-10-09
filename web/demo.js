import { loadHeadlineGPT } from "./src/index.js";

const $ = (id) => document.getElementById(id);
const status = (s) => ($("status").textContent = s);

async function main() {
  if (!navigator.gpu) return status("WebGPU is not available in this browser.");
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  const device = await adapter.requestDevice();
  device.lost.then((info) => status(`GPU device lost: ${info.message}`));

  const t0 = performance.now();
  const { gpt, tokenizer, meta } = await loadHeadlineGPT("./public/headline-gpt.bin", device, { maxRows: 512 });
  const pretrained = await gpt.readWeights();
  status(`${(gpt.nParams / 1e6).toFixed(2)}M params · val loss ${meta.val_loss_int8 ?? "?"} · loaded in ${(performance.now() - t0).toFixed(0)} ms`);

  const sample = async () => {
    const t = performance.now();
    const rows = await gpt.generate({ n: 8, temperature: +$("temp").value, topK: 50 });
    $("samples").innerHTML = "";
    for (const ids of rows) $("samples").append(Object.assign(document.createElement("li"), { textContent: tokenizer.decode(ids) }));
    $("log").textContent = `sampled 8 in ${(performance.now() - t).toFixed(0)} ms`;
  };

  const train = async () => {
    const ids = [0, ...tokenizer.encode($("target").value), 0];
    const T = ids.length;
    const g = new Float32Array(T).map((_, t) => (t < T - 1 ? -1 / (T - 1) : 0)); // mean NLL
    const lines = [];
    let total = 0;
    for (let step = 0; step < 20; step++) {
      const t = performance.now();
      const logp = await gpt.trainStep(ids, 1, T, g, { lr: 3e-4 }, { readLogp: true });
      total += performance.now() - t;
      const nll = -logp.slice(0, T - 1).reduce((a, b) => a + b, 0) / (T - 1);
      if (step % 4 === 0 || step === 19) lines.push(`step ${String(step).padStart(2)}  nll/token ${nll.toFixed(3)}`);
    }
    lines.push(`${(total / 20).toFixed(1)} ms per step (fwd + bwd + AdamW + readback)`);
    $("log").textContent = lines.join("\n");
  };

  $("gen").onclick = sample;
  $("train").onclick = train;
  $("reset").onclick = () => { gpt.setWeights(pretrained); $("log").textContent = "weights reset to pretrained"; };
  for (const b of ["gen", "train", "reset"]) $(b).disabled = false;
  await sample();
}

main().catch((e) => { status(`error: ${e.message}`); console.error(e); });
