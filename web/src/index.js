import { GPT } from "./gpt.js";
import { parseModel } from "./format.js";
import { Tokenizer } from "./tokenizer.js";
import { DPOSession } from "./rlhf.js";

export { GPT, parseModel, Tokenizer, DPOSession };
export { sampleLogits } from "./gpt.js";

export class WebGPUUnavailableError extends Error {
  name = "WebGPUUnavailableError";
}

/** Fetch a .bin, build the tokenizer and the GPU model. */
export async function loadHeadlineGPT(url, device, opts) {
  const buf = await (await fetch(url)).arrayBuffer();
  const { config, tokenizer, tensors, meta } = parseModel(buf);
  return { gpt: new GPT(device, config, tensors, opts), tokenizer: new Tokenizer(tokenizer), meta };
}

/** Download with progress callbacks (loaded, total bytes). */
export async function fetchBytes(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`could not load ${url}: ${res.status}`);
  const total = Number(res.headers.get("content-length")) || 0;
  if (!res.body || !onProgress) return res.arrayBuffer();
  const reader = res.body.getReader();
  const chunks = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    onProgress(loaded, total);
  }
  const out = new Uint8Array(loaded);
  let off = 0;
  for (const c of chunks) { out.set(c, off); off += c.length; }
  return out.buffer;
}

/**
 * Browser entry point: get a WebGPU device, download the model, build a fresh DPO session
 * (trainable policy + frozen reference). Nothing is shared between page loads.
 */
export async function loadDPOSession(url, { onProgress, ...opts } = {}) {
  if (!globalThis.navigator?.gpu) throw new WebGPUUnavailableError("WebGPU is not available in this browser");
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new WebGPUUnavailableError("No WebGPU adapter found");
  const device = await adapter.requestDevice();
  const bytes = await fetchBytes(url, onProgress);
  const session = DPOSession.fromBytes(device, bytes, opts);
  session.device = device;
  return session;
}
