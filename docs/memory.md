# Memory and speed

How `oldiron fit` arrives at its numbers, so you can tell which parts are exact and which
are estimates.

## The budget

Four things have to fit in GPU memory at once.

**Weights** are exact: the size of the GGUF file, or the sum of every shard for a
multi-part model.

**KV cache** is computed from the model's own metadata rather than a rule of thumb:

```
bytes = ctx x sum over layers of  kv_heads x (head_dim_k x bytes_k + head_dim_v x bytes_v)
```

The sum runs per layer, not layers x a single number, because hybrid architectures have
layers with no attention cache at all — counting those would inflate the estimate badly.
Quantized cache types use their real block sizes, so `q8_0` is 34 bytes per 32 values,
not 32.

**Compute buffers** scale with the physical batch size (`-ub`) and the model width, plus
the logits buffer, which depends on vocabulary size. This is an estimate.

**Runtime overhead** is the driver context and a safety margin: a few hundred MiB per
CUDA device, less on Metal, where there is no comparable per-device context and the OS
headroom is already excluded by the allocation cap.

The first two are exact; the last two are why the footer says the total is within about
15%. `llama-server --fit` does the final adjustment at load time — oldiron answers the
question you have before the model is on your disk.

## Choosing a plan

Given a target context, oldiron tries KV cache types in order — `f16/f16`, then
`q8_0/q8_0`, then `q8_0/q4_0` — and picks the first that fits. Quantized KV requires
Flash Attention, so the generated command always carries `-fa on`.

If nothing fits with the whole model on the GPU, it computes how many layers do fit and
sets `-ngl` to that number, with a warning that the rest lands on the CPU. For
mixture-of-experts models it suggests `-cmoe` / `-ncmoe` instead, since keeping expert
weights on the CPU costs less speed than moving whole layers.

## The speed ceiling

Generating one token reads every active weight once, so:

```
tokens/s ceiling = memory bandwidth / active weight bytes
```

A 4.7 GB model on a 100 GB/s M2 cannot exceed about 21 tokens/s no matter what else is
tuned. Treat it as a hard upper bound rather than a prediction: how close a real run gets
depends on the backend and the architecture, and older cards without tensor cores fall
much further short than recent ones. It still answers the question that matters — whether
a setup can possibly be pleasant, or is merely possible.

Two caveats oldiron states explicitly:

- For MoE models only a fraction of the weights is read per token, so the real figure is
  well above this ceiling.
- When layers sit on the CPU, the ceiling no longer applies at all — the PCIe bus and CPU
  memory dominate, and the result is far slower.

## Reading the numbers back

`oldiron info` prints the metadata behind all of this, including the KV cost per token,
and `--json` on either command gives the byte-level figures for scripting.
