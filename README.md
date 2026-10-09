# In-browser clickbait rlhf

I wanted to demonstrate RLHF in real-time; the user A/B selects and watches the model re-rank instantly
The goal is to "pretrain a few-million-parameter GPT offline on a big pile of boring headlines (Kaggle’s “A Million News Headlines” from ABC Australia works), then ship it at a few MB with the training code written in TF.js or raw WebGPU."

## What's here

| path | what |
|---|---|
| `pretrain/` | PyTorch (MPS) data prep, BPE tokenizer, pretraining, int8 export |
| `web/src/` | The browser model: hand-written WGSL forward + backward + AdamW (`kernels.js`, `gpt.js`), tokenizer, loader |
| `web/public/headline-gpt.bin` | The shipped model: weights (int8, per-row scales) + tokenizer in one file |
| `web/index.html`, `web/demo.js` | Demo page: sample headlines, run in-browser training steps |
| `web/test/` | Parity test against PyTorch, benchmark (run in Node via Dawn's `webgpu` package) |

### Model

GPT-2 style, pre-LayerNorm, tanh GELU, tied embeddings: 4 layers, d=256, 8 heads, 32-token context,
4096-token BPE vocab trained on the headlines (lowercase ASCII, `▁` word prefix). 4.22M parameters.

### Why raw WebGPU, not TF.js

Same model, same train step, same idle M3 Pro GPU (`npm run bench` vs an equivalent TF.js graph):

| train step (fwd + bwd + AdamW) | raw WebGPU | TF.js 4.22 webgpu backend | speedup |
|---|---|---|---|
| B=2, T=24 (one A/B pair) | 5.1 ms | 47.1 ms | 9.3× |
| B=8, T=24 | 8.2 ms | 52.1 ms | 6.3× |
| B=16, T=32 | 17.3 ms | 83.8 ms | 4.8× |
| B=64, T=32 | 58.7 ms | 196.6 ms | 3.4× |

In headless Chrome (Metal): 4.6 ms per batch-1 train step, ~50 ms to sample 8 headlines, 0.6 s to load.
Pretrained val loss 3.947 nats/token (int8 costs +0.0003). File: 4.46 MB, 4.0 MB gzipped.

The gap is largest at the tiny batches an interactive RLHF loop uses: TF.js dispatches one generic
shader per op (plus autograd bookkeeping), while `gpt.js` records each (batch, length) shape once as a
fixed list of fused, size-tuned dispatches (split-K matmuls for small batches, one workgroup per
attention head) and replays it with a single `submit`. TF.js has also not had a release since 4.22.

## Rebuilding

```sh
# 1. data (Harvard Dataverse mirror of the Kaggle dataset; no Kaggle login needed)
mkdir -p data/raw
curl -L -o data/raw/abcnews-date-text.tsv "https://dataverse.harvard.edu/api/access/datafile/6329050?format=original"

# 2. tokenizer + tokenized arrays (~15 s)
cd pretrain
uv run python prepare.py ../data/raw/abcnews-date-text.tsv ../data

# 3. pretrain on the M-series GPU (~27 min for 20k steps)
uv run python train.py --steps 20000 --out ../data/ckpt.pt

# 4. quantize + ship, and write the parity fixture
uv run python export.py --ckpt ../data/ckpt.pt --out ../web/public/headline-gpt.bin --fixture ../data/fixture.bin

# 5. verify the WebGPU implementation against PyTorch, and benchmark
cd ../web && npm install
npm test     # logprobs, all 52 gradient tensors, and one AdamW step vs float64 PyTorch
npm run bench

# 6. demo
python3 -m http.server 8765    # then open http://localhost:8765/ (Chrome, Edge, or Safari 26+)
```

## Browser API

```js
import { loadHeadlineGPT } from "./src/index.js";

const device = await (await navigator.gpu.requestAdapter()).requestDevice();
const { gpt, tokenizer } = await loadHeadlineGPT("./public/headline-gpt.bin", device, { maxRows: 512 });

const rows = await gpt.generate({ n: 8, temperature: 0.9, topK: 50 });   // token arrays
rows.map((ids) => tokenizer.decode(ids));

// Any objective expressed as dLoss/dlogp per token:
const logp = await gpt.forward(tokens, B, T);   // logp[b*T+t] = log p(tokens[b,t+1] | prefix)
await gpt.backward(gradLogp, { lr: 1e-4 });     // backprop + clipped AdamW step
// or, when gradLogp doesn't depend on the forward (e.g. plain LM loss), one fused submit:
await gpt.trainStep(tokens, B, T, gradLogp, { lr: 1e-4 });

const ref = await gpt.readWeights();   // snapshot, e.g. the frozen reference policy
gpt.setWeights(ref);                   // restore + reset optimizer state
```

Rows are `[<|eot|>, w1..wn, <|eot|>, pad…]` (token 0 is `<|eot|>`; pad with 0 and give those
positions zero gradient). `forward` → `backward` is the shape a DPO step takes: read the sequence
log-probs, compute the per-sequence coefficient on the CPU, send it back as `gradLogp`.
