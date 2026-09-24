# Hardware notes

This is the knowledge `oldiron doctor` applies. It is short on purpose: these are the
few facts that decide whether an old card works at all.

## The 2025-2026 cutoff for old NVIDIA cards

Two separate deprecations landed close together, and they bite in different ways.

**CUDA 13.0 removed Maxwell, Pascal and Volta** — everything below compute capability
7.5. A CUDA 13 toolkit either refuses to build for these architectures or produces a
binary the card cannot run.

**Driver branch 580 is the last one that supports them.** Installing a 590-series driver
on a Pascal card can leave it not enumerating at all. Distributions that moved to 590 by
default usually keep a legacy package around (`nvidia-580xx-dkms` on Arch, for example).

So for a P40, P100, M40 or V100 the working combination is **CUDA 12.x with driver 580 or
older**, and the thing to guard against is an unattended system upgrade.

| Architecture | Compute capability | Last CUDA | Last driver |
| --- | --- | --- | --- |
| Kepler | 3.5, 3.7 | 11.8 | 470 |
| Maxwell | 5.0, 5.2 | 12.9 | 580 |
| Pascal | 6.0, 6.1 | 12.9 | 580 |
| Volta | 7.0 | 12.9 | 580 |
| Turing and newer | 7.5+ | current | current |

## Pascal is two different cards

The P100 (GP100, cc 6.0) has full-rate FP16 — twice FP32 — and HBM2 at 732 GB/s. The P40
and the GTX 10-series (GP102/GP104, cc 6.1) run FP16 arithmetic at **1/64 rate**. On
those, FP16 cuBLAS paths are a trap: building with `-DGGML_CUDA_FORCE_MMQ=ON` keeps
matrix multiplication on the integer kernels, which is both faster and lighter on VRAM.

`oldiron doctor` adds that flag only for cc 6.1, not for the P100.

## CPUs without AVX2

A Sandy Bridge or Ivy Bridge Xeon has AVX but not AVX2 or FMA. Most prebuilt binaries and
most PyPI wheels containing native code are compiled for AVX2, and on these CPUs they die
with `SIGILL` — an illegal instruction — often from a transitive dependency you never
chose. The fixes are to build llama.cpp from source with `-DGGML_NATIVE=ON`, and to use
`pip install --no-binary :all:` for native packages that crash.

This is also why oldiron itself has no dependencies: it has to install on the machine
that is having the problem.

## Apple Silicon

Metal does not get "VRAM" — it gets a share of unified memory. The default ceiling is
roughly **75% of total RAM**, so a 16 GB Mac offers about 12 GB to the GPU, and the model
plus KV cache must fit under that, not under the full figure. It can be raised:

```bash
sudo sysctl iogpu.wired_limit_mb=N
```

Leave several GiB for macOS; setting it near 100% causes memory pressure rather than
speed. `oldiron` reads the current value if you have set one.

The other Apple-specific detail is core layout: these chips have performance and
efficiency cores, and giving llama.cpp every core usually makes it slower. `oldiron`
recommends `-t` with the performance-core count.

A note on feature detection, since it caused a real bug here: macOS spells these sysctl
keys the way Arm does, so it is `hw.optional.arm.FEAT_DotProd` in mixed case while
`FEAT_I8MM` and `FEAT_BF16` are not, and sysctl keys are case-sensitive.

## Memory bandwidth by device

Decode reads every weight once per generated token, so bandwidth sets the speed ceiling
(see [Memory and speed](memory.md)).

| Device | Bandwidth |
| --- | --- |
| Tesla P100 | 732 GB/s |
| Tesla P40 | 347 GB/s |
| Tesla M40 | 288 GB/s |
| Apple M1 / M2 / M3 | 68 / 100 / 100 GB/s |
| Apple M4 | 120 GB/s |
| Apple M1-M3 Max | 400 GB/s |
| Apple M4 Max | 546 GB/s |

If your device is missing or wrong, `oldiron doctor --json` output attached to an issue
is enough to add it.
