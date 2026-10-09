import { GPT } from "./gpt.js";
import { parseModel } from "./format.js";
import { Tokenizer } from "./tokenizer.js";

export { GPT, parseModel, Tokenizer };
export { sampleLogits } from "./gpt.js";

/** Fetch a .bin, build the tokenizer and the GPU model. */
export async function loadHeadlineGPT(url, device, opts) {
  const buf = await (await fetch(url)).arrayBuffer();
  const { config, tokenizer, tensors, meta } = parseModel(buf);
  return { gpt: new GPT(device, config, tensors, opts), tokenizer: new Tokenizer(tokenizer), meta };
}
