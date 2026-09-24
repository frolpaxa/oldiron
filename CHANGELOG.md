# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2026-09-24

### Fixed

- Restore the `License :: OSI Approved :: MIT License` classifier, dropped while reverting
  the PEP 639 license field. PyPI already showed the licence, but the classifier is what
  licence-based search filters use.
- Point the PyPI badges at `.svg` URLs. GitHub's image proxy had cached the "not found"
  responses from before the package was published, and a new URL is the only way to get it
  to look again.

## [0.1.0] - 2026-09-23

First release.

### Added

- `oldiron doctor`: detects NVIDIA GPUs, CPU instruction sets, RAM, driver and CUDA
  toolkit versions, then prints findings and a CMake recipe tailored to the machine.
- Knowledge base for legacy NVIDIA architectures (Kepler through Volta): the last
  supporting CUDA toolkit and driver branch, FP16 throughput characteristics, and memory
  bandwidth. Flags the combinations that cannot work, such as CUDA 13.x or driver branch
  590 with a Pascal card.
- `-DGGML_CUDA_FORCE_MMQ=ON` is recommended for compute capability 6.1 only, where FP16
  arithmetic runs at 1/64 rate, and not for the P100.
- Warning for CPUs without AVX2, where prebuilt binaries and native PyPI wheels fail with
  `SIGILL`.
- `oldiron fit`: computes the memory budget for a GGUF model — weights, KV cache, compute
  buffers and runtime overhead — against what the GPU can actually allocate, then prints a
  ready `llama-server` command with `-ngl`, `-c` and KV cache types chosen to fit. The
  budget always describes the plan being recommended: under a partial offload only the
  offloaded weights count against VRAM, and the remainder is reported as host RAM. When a
  shorter context would fit entirely on the GPU, it says which one.
- Dependency-free GGUF metadata reader. KV cache is summed per layer, so hybrid models
  whose layers hold no attention cache are measured correctly.
- `oldiron info`: prints the model metadata the arithmetic is based on.
- `oldiron bench`: runs `llama-bench` with the settings `fit` recommends, reports measured
  prompt and generation speed against the bandwidth ceiling, and writes a record that can
  be contributed to a shared results table.
- macOS and Apple Silicon support: detection through `sysctl`, budgeting against the
  Metal allocation cap (about 75% of unified memory) rather than total RAM, a Metal build
  recipe, and a `-t` recommendation based on performance-core count.
- Decode speed ceiling derived from memory bandwidth, with explicit caveats for MoE models
  and for partial CPU offload.
- Reading GGUF metadata directly from Hugging Face over HTTP range requests, so a model
  can be checked before downloading it. Accepts `hf:<user>/<repo>[:QUANT]`, a revision
  suffix (`@refs/pr/3`) and Hub blob URLs; sums sharded models across every part and
  reads `HF_TOKEN` for gated repositories.
- Optional `rich` output — panels, a segmented memory bar and colour — installed with
  `pip install "oldiron[rich]"`. Without it the output degrades to plain text; `--plain`
  forces that layout, and `--json` is unaffected by either.
- Documentation on Read the Docs, and publication to PyPI through trusted publishing.

[Unreleased]: https://github.com/frolpaxa/oldiron/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/frolpaxa/oldiron/releases/tag/v0.1.1
[0.1.0]: https://github.com/frolpaxa/oldiron/releases/tag/v0.1.0
