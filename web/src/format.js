// Reads the single-file model format written by pretrain/export.py:
//   "HGPT" | u32 headerBytes | JSON header | pad to 4 | tensor data
// int8 tensors carry one f32 scale per row (w = q * scale[row]).

export function parseModel(arrayBuffer) {
  const view = new DataView(arrayBuffer);
  const magic = new TextDecoder().decode(new Uint8Array(arrayBuffer, 0, 4));
  if (magic !== "HGPT") throw new Error(`not a headline-gpt file (magic ${magic})`);
  const headerBytes = view.getUint32(4, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(arrayBuffer, 8, headerBytes)));
  const base = 8 + Math.ceil(headerBytes / 4) * 4;
  const tensors = {};
  for (const t of header.tensors) {
    const n = t.shape.reduce((a, b) => a * b, 1);
    if (t.dtype === "f32") {
      tensors[t.name] = new Float32Array(arrayBuffer.slice(base + t.offset, base + t.offset + n * 4));
    } else if (t.dtype === "int8") {
      const q = new Int8Array(arrayBuffer, base + t.offset, n);
      const rows = t.shape[0], cols = n / rows;
      const scale = new Float32Array(arrayBuffer.slice(base + t.scaleOffset, base + t.scaleOffset + rows * 4));
      const out = new Float32Array(n);
      for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) out[r * cols + c] = q[r * cols + c] * scale[r];
      tensors[t.name] = out;
    } else {
      throw new Error(`unknown dtype ${t.dtype}`);
    }
  }
  return { config: header.config, tokenizer: header.tokenizer, tensors, meta: header.meta ?? {} };
}

/** Same container, f32 only: used for test fixtures. */
export function parseBlob(arrayBuffer) {
  return parseModel(arrayBuffer);
}
