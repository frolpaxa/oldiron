# oldiron

**Run local LLMs on the hardware you already have.**

```
Tesla P100-PCIE-16GB  +  Qwen3 8B Q4_K_M
       16 GiB              4.68 GiB weights
                           -> fits, ~15 tok/s ceiling
```

A second-hand Tesla P40 costs less than a game. The problem is never the card — it is
the hour you lose to a build that segfaults, a driver that no longer sees the GPU, a
wheel compiled for AVX2 on a CPU that does not have it, and a 15 GB download that turns
out not to fit anyway.

`oldiron` is one command that looks at your machine and tells you what will work before
you spend that hour.

## Quick start

```bash
pip install oldiron           # zero dependencies
pip install "oldiron[rich]"   # same, plus colour, panels and a memory bar
```

```bash
# What is this machine, and how should llama.cpp be built for it?
oldiron doctor

# Does this model fit, and how do I launch it?
oldiron fit ~/models/Qwen3-8B-Q4_K_M.gguf -c 16384

# Same question, before downloading 5 GB
oldiron fit hf:unsloth/Qwen3-8B-GGUF:Q4_K_M
```

## What it does

- **Knows about old cards.** Kepler, Maxwell, Pascal, Volta: which CUDA toolkit still
  supports them, which driver branch is the last one, which build flags to use, and
  which FP16 paths to avoid.
- **Budgets memory honestly.** Weights and KV cache are computed from the file's own
  metadata, per layer, including hybrid models where only some layers hold a cache.
- **Says whether it will be usable, not merely loadable.** Decode speed is bounded by
  memory bandwidth, and that ceiling is printed next to the memory budget.
- **Answers before the download.** Point it at a Hugging Face repo and it reads the
  GGUF header over HTTP range requests.
- **Speaks CUDA and Metal.** On Apple Silicon it budgets against the Metal allocation
  cap rather than total RAM, and recommends performance-core counts.

## Zero dependencies, on purpose

The machines this tool exists for are the ones where installing things goes wrong. The
core has no dependencies at all, and `rich` is an optional extra: without it the output
degrades to plain text rather than failing.

```{toctree}
:maxdepth: 2

cli
hardware
memory
api
changelog
```
