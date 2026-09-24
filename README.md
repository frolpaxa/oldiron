# oldiron

[![PyPI](https://img.shields.io/pypi/v/oldiron.svg)](https://pypi.org/project/oldiron/)
[![Python](https://img.shields.io/pypi/pyversions/oldiron.svg)](https://pypi.org/project/oldiron/)
[![Docs](https://readthedocs.org/projects/oldiron/badge/?version=latest)](https://oldiron.readthedocs.io)
[![Tests](https://github.com/frolpaxa/oldiron/actions/workflows/ci.yml/badge.svg)](https://github.com/frolpaxa/oldiron/actions/workflows/ci.yml)
[![License](https://img.shields.io/pypi/l/oldiron.svg)](LICENSE)

**Run local LLMs on the hardware you already have.**

Documentation: [oldiron.readthedocs.io](https://oldiron.readthedocs.io)

A second-hand Tesla P40 costs less than a game. The problem is never the card — it is
the hour you lose to a build that segfaults, a driver that no longer sees the GPU, a
wheel compiled for AVX2 on a CPU that does not have it, and a 15 GB download that turns
out not to fit anyway.

`oldiron` is a single command that looks at your machine and tells you what will work
before you spend the hour.

- **Zero dependencies, pure Python.** It installs on the machine that is the problem.
  Nothing here is compiled, so nothing here can die with `SIGILL`.
- **Knows about old cards.** Kepler, Maxwell, Pascal, Volta — which CUDA toolkit still
  supports them, which driver branch is the last one, which architecture flags to build
  with, and which FP16 paths to avoid.
- **Tells you what fits before you download.** Reads GGUF metadata and does the KV-cache
  arithmetic per layer, including hybrid models where only some layers hold a cache.
- **Tells you whether it will be usable, not just loadable.** Decode speed is bounded by
  memory bandwidth, so it prints that ceiling alongside the memory budget.
- **Linux/CUDA and macOS/Metal.** On Apple Silicon it budgets against the Metal
  allocation cap (about 75% of unified memory) rather than total RAM, and recommends
  performance-core counts instead of every core.

## Install

```bash
pip install oldiron           # zero dependencies
pip install "oldiron[rich]"   # same, plus colour, panels and a memory bar
```

Needs Python 3.8+ and nothing else. `rich` is optional on purpose: the whole point is that
this installs on the machine that is giving you trouble, so without it the output degrades
to plain text rather than failing. `--plain` forces the plain layout even when rich is
present, and `--json` is unaffected by either. Editable installs (`pip install -e .`) need
pip 21.3 or newer, since the project ships only a `pyproject.toml`.

## `oldiron doctor`

```
$ oldiron doctor
System   Linux 6.1.0-28-amd64
CPU      Intel(R) Xeon(R) CPU E3-1230 V2 @ 3.30GHz (8 threads, SIMD: avx)
RAM      31.30 GiB
GPU 0    Tesla P100-PCIE-16GB -- 16.00 GiB (Pascal, cc 6.0)
Driver   580.65.06        CUDA: 12.4

[ ok ] Tesla P100-PCIE-16GB
       GP100 has full-rate FP16 (2x FP32) and HBM2 -- unusual for Pascal.
[ ok ] Legacy GPU detected
       Tesla P100-PCIE-16GB (Pascal, cc 6.0). CUDA 13.0 removed support for Maxwell,
       Pascal and Volta, and driver branch 580 is the last one that supports them.
       fix: Stay on CUDA 12.x and driver 580 or older. Do not let a distro upgrade
       pull in 590+.
[warn] CPU without AVX2 (Intel(R) Xeon(R) CPU E3-1230 V2 @ 3.30GHz)
       Most prebuilt binaries and PyPI wheels are compiled for AVX2 and die with
       SIGILL (illegal instruction) on this CPU. This includes many Rust/C++ wheels
       pulled in as transitive dependencies.
       fix: Build llama.cpp from source on this machine with -DGGML_NATIVE=ON, and
       prefer pip install --no-binary :all: for native packages that crash.

Build recipe for this machine:

  git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
  cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="60" -DGGML_CUDA_FA_ALL_QUANTS=ON -DGGML_NATIVE=ON
  cmake --build build --config Release -j 8
```

Put a driver from branch 590 in front of that same card and the report turns red
instead, before you have rebuilt anything.

## `oldiron fit`

```
$ oldiron fit models/Qwen3-27B-UD-IQ4_XS.gguf -c 8192
Model    Qwen3 27B  [qwen3, IQ4_XS]
         64 layers, trained context 262144, 1 file(s)
GPU 0    Tesla P100-PCIE-16GB -- 16.00 GiB (Pascal, cc 6.0)

Plan     context 8192, KV cache q8_0/q4_0, -ngl 58

  weights             14.63 GiB
  KV cache              832 MiB
  compute buffers       381 MiB  (estimate)
  CUDA overhead         912 MiB  (estimate)
  --------------------------------
  total               16.71 GiB
  available VRAM      16.00 GiB
  short by              727 MiB  (DOES NOT FIT)

Max context by KV cache type (fully offloaded):
  f16/f16            1024 tokens
  q8_0/q8_0          2048 tokens
  q8_0/q4_0          2048 tokens
  q4_0/q4_0          4096 tokens

Start with:

  llama-server -m models/Qwen3-27B-UD-IQ4_XS.gguf -c 8192 -ngl 58 -fa on -ctk q8_0 -ctv q4_0 --host 127.0.0.1 --port 8080
```

On an Apple Silicon Mac the same commands speak Metal instead:

```
$ oldiron doctor
System   Darwin 23.6.0
CPU      Apple M1 (8 threads, SIMD: neon + dotprod, i8mm, bf16, fp16)
RAM      8.00 GiB
GPU 0    Apple M1 -- 6.00 GiB shared (Apple Silicon, 8-core GPU)  68 GB/s
Backend  Metal (unified memory)

[ ok ] Heterogeneous cores
       4 performance and 4 efficiency cores. Handing llama.cpp every core usually
       makes it slower, not faster.
       fix: Pass -t 4 so only the performance cores are used.
[ ok ] Unified memory
       CPU and GPU share 8.00 GiB. Metal will allocate up to 6.00 GiB of it (about
       75% by default), so weights plus KV cache must fit under that, not under the
       full RAM figure.
       fix: Raise it if needed: sudo sysctl iogpu.wired_limit_mb=N -- but leave
       several GiB for macOS itself.
[warn] Tight memory for local models
       8.00 GiB total means roughly 6.00 GiB usable by the GPU. Expect 4B-8B models
       at Q4, not larger ones.

Build recipe for this machine:

  xcode-select --install   # once, for the command line tools
  git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
  cmake -B build -DGGML_METAL=ON -DGGML_CPU_KLEIDIAI=ON
  cmake --build build --config Release -j 4
```

## Checking before you download

A GGUF file keeps its metadata at the front, so a couple of HTTP range requests answer
"will this fit?" without pulling the weights. Point `fit` at a repo instead of a path:

```
$ oldiron fit hf:unsloth/Qwen3-4B-GGUF:Q4_K_M -c 16384
Model    Qwen3 4B  [qwen3, Q4_K_M]
         36 layers, trained context 40960, 1 file(s)
         unsloth/Qwen3-4B-GGUF:Q4_K_M on Hugging Face (metadata only, not downloaded)
...
Start with:

  llama-server -hf unsloth/Qwen3-4B-GGUF:Q4_K_M -c 16384 -ngl all -fa on -ctk q8_0 -ctv q4_0 -t 4 --host 127.0.0.1 --port 8080
```

Two requests and about 3 MB settle a 2.3 GB question, and the command it prints hands the
download to `llama-server` itself. Sharded models are summed across every part, so the
weight total is the real one. Omit the quant to get llama.cpp's own `Q4_K_M` default; name
a revision with `@`, as in `hf:user/repo:Q4_K_M@refs/pr/3`; a blob URL copied from the Hub
works too. Gated repos read `HF_TOKEN` (or `--hf-token`).

`oldiron info model.gguf` prints the metadata the arithmetic is based on, and accepts the
same `hf:` specs. Every command takes `--json` if you would rather script it.

## `oldiron bench`

`fit` predicts; `bench` checks. It runs `llama-bench` with the settings `fit` recommends
and puts the measurement next to the ceiling:

```
  prompt processing     465.13  tok/s  +-5.02
  generation             11.42  tok/s  +-0.18
  bandwidth ceiling      55.70  tok/s  theoretical

  21% of the ceiling  — what this card actually delivers
```

How close a card gets to its bandwidth ceiling is an empirical fact, and for old hardware
it is largely unpublished. Each run writes a JSON record into `results/`; sending it in as
a pull request is what builds the table.

## What it is not

It does not replace `llama-server --fit`, which sizes the context at load time — it
answers the question you have *before* the model is on your disk, and it is the part
that knows your driver branch is about to stop supporting your card.

Compute buffer and CUDA context sizes are estimates, within roughly 15%. Weights and KV
cache are computed exactly from the file and its metadata.

## Roadmap

- A `bench` command that records tokens/s per flag combination, building a public table
  of what old cards actually do.
- AMD (ROCm/HIP) and Vulkan paths for equally cheap second-hand Radeons.

Contributions with real hardware data are the most useful kind. If your card is not in
`gpudb.py`, open an issue with the output of `oldiron doctor --json`.

## License

MIT.
