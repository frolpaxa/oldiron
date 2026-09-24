"""Turn detected hardware into findings, a build recipe and a run command."""

from __future__ import annotations

from dataclasses import dataclass

from . import memory as mem
from .gguf_meta import GgufModel
from .gpudb import arch_for, is_legacy
from .hardware import System
from .memory import human

OK, WARN, ERROR = "ok", "warn", "error"


@dataclass
class Finding:
    level: str
    title: str
    detail: str
    fix: str = ""


def _major(version: str | None) -> int:
    try:
        return int(str(version).split(".")[0])
    except (ValueError, AttributeError):
        return 0


def check_system(system: System) -> list[Finding]:
    findings: list[Finding] = []

    if not system.gpus:
        findings.append(Finding(
            WARN, "No NVIDIA GPU detected",
            "nvidia-smi returned nothing, so this is a CPU-only plan.",
            "If you do have a card, check that the driver is loaded: nvidia-smi -L",
        ))

    legacy_names: list[str] = []
    for gpu in system.gpus:
        if gpu.unified:  # integrated GPU, no compute capability to look up
            continue
        info = arch_for(gpu.compute_cap, gpu.name)
        if info is None:
            findings.append(Finding(
                WARN, f"Unknown GPU: {gpu.name}",
                "Compute capability is not in the database; advice falls back to generic defaults.",
                "Open an issue with the output of: nvidia-smi --query-gpu=name,compute_cap --format=csv",
            ))
            continue
        if is_legacy(gpu.compute_cap):
            legacy_names.append(f"{gpu.name} ({info.name}, cc {info.cc})")
        for note in info.notes:
            findings.append(Finding(OK, f"{gpu.name}", note))

    if legacy_names:
        findings.append(Finding(
            OK, "Legacy GPU detected",
            ", ".join(legacy_names) + ". CUDA 13.0 removed support for Maxwell, Pascal and "
            "Volta, and driver branch 580 is the last one that supports them.",
            "Stay on CUDA 12.x and driver 580 or older. Do not let a distro upgrade pull in 590+.",
        ))

    driver_major = _major(system.driver_version)
    if legacy_names and driver_major >= 590:
        findings.append(Finding(
            ERROR, "Driver too new for this GPU",
            f"Driver {system.driver_version} is past branch 580, the last one supporting this "
            "architecture. The card may fail to enumerate.",
            "Install the legacy 580 branch (on Arch: nvidia-580xx-dkms; on Debian/Ubuntu: the "
            "580 package from your distro or NVIDIA's legacy archive).",
        ))

    cuda_major = _major(system.cuda_version)
    if legacy_names and cuda_major >= 13:
        findings.append(Finding(
            ERROR, "CUDA toolkit too new",
            f"nvcc reports CUDA {system.cuda_version}; 13.0 dropped these architectures, so the "
            "build will either fail or produce a binary your card cannot run.",
            "Install CUDA 12.x and point cmake at it: -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.x/bin/nvcc",
        ))
    elif legacy_names and cuda_major == 0:
        findings.append(Finding(
            WARN, "CUDA toolkit not found",
            "nvcc is not on PATH, so llama.cpp cannot be built with CUDA support.",
            "Install the CUDA 12.x toolkit (not 13.x) for this GPU.",
        ))

    cpu = system.cpu
    if cpu.is_arm:
        # AVX is an x86 instruction set; warning about it on Arm is meaningless.
        if not cpu.has("FEAT_DotProd"):
            findings.append(Finding(
                WARN, "Arm CPU without dotprod",
                "No FEAT_DotProd, so the quantized CPU kernels fall back to slower paths.",
            ))
        if cpu.perf_cores and cpu.perf_cores < cpu.cores:
            findings.append(Finding(
                OK, "Heterogeneous cores",
                f"{cpu.perf_cores} performance and {cpu.cores - cpu.perf_cores} efficiency cores. "
                "Handing llama.cpp every core usually makes it slower, not faster.",
                f"Pass -t {cpu.perf_cores} so only the performance cores are used.",
            ))
    else:
        if not cpu.has("avx2"):
            findings.append(Finding(
                WARN, f"CPU without AVX2 ({cpu.model.strip() or 'unknown'})",
                "Most prebuilt binaries and PyPI wheels are compiled for AVX2 and die with "
                "SIGILL (illegal instruction) on this CPU. This includes many Rust/C++ wheels "
                "pulled in as transitive dependencies.",
                "Build llama.cpp from source on this machine with -DGGML_NATIVE=ON, and prefer "
                "pip install --no-binary :all: for native packages that crash.",
            ))
        if not cpu.has("avx"):
            findings.append(Finding(
                WARN, "CPU without AVX",
                "CPU-side prompt processing will be slow; keep as much of the model on the "
                "GPU as possible.",
            ))

    if system.unified_memory:
        gpu = system.gpus[0]
        findings.append(Finding(
            OK, "Unified memory",
            f"CPU and GPU share {human(system.ram_bytes)}. Metal will allocate up to "
            f"{human(gpu.vram_bytes)} of it (about 75% by default), so weights plus KV cache "
            "must fit under that, not under the full RAM figure.",
            "Raise it if needed: sudo sysctl iogpu.wired_limit_mb=N -- but leave several GiB "
            "for macOS itself.",
        ))
        if system.ram_bytes and system.ram_bytes < 16 * mem.GIB:
            findings.append(Finding(
                WARN, "Tight memory for local models",
                f"{human(system.ram_bytes)} total means roughly {human(gpu.vram_bytes)} usable "
                "by the GPU. Expect 4B-8B models at Q4, not larger ones.",
            ))

    if (system.ram_bytes and system.gpus and not system.unified_memory
            and system.ram_bytes < system.total_vram_bytes):
        findings.append(Finding(
            WARN, "Less RAM than VRAM",
            f"RAM {mem.human(system.ram_bytes)} vs VRAM {mem.human(system.total_vram_bytes)}. "
            "Loading a model that fills VRAM may thrash or fail while mmapping from disk.",
            "Use --load-mode mmap (the default) and avoid mlock.",
        ))

    caps = {g.compute_cap for g in system.gpus if g.compute_cap}
    if len(caps) > 1:
        findings.append(Finding(
            WARN, "Mixed GPU architectures",
            f"Compute capabilities {sorted(caps)} in one machine. The build must target all of them "
            "and the slowest card sets the pace.",
            "Pass every architecture to CMAKE_CUDA_ARCHITECTURES and split by layer with -sm layer.",
        ))
    return findings


def build_recipe(system: System) -> list[str]:
    """CMake invocation tailored to the detected hardware."""
    if system.backend == "metal":
        # Metal and the Accelerate BLAS are on by default on macOS; naming Metal
        # explicitly makes the failure obvious if the SDK is missing.
        flags = ["-DGGML_METAL=ON"]
        if system.cpu.has("FEAT_I8MM") or system.cpu.has("FEAT_DotProd"):
            flags.append("-DGGML_CPU_KLEIDIAI=ON")
        jobs = max(1, system.cpu.threads_hint)
        return [
            "xcode-select --install   # once, for the command line tools",
            "git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp",
            "cmake -B build " + " ".join(flags),
            f"cmake --build build --config Release -j {jobs}",
        ]

    if not system.gpus:
        flags = ["-DGGML_CUDA=OFF"]
    else:
        archs = sorted({g.compute_cap.replace(".", "") for g in system.gpus if g.compute_cap})
        flags = ["-DGGML_CUDA=ON"]
        if archs:
            flags.append(f'-DCMAKE_CUDA_ARCHITECTURES="{";".join(archs)}"')
        # Pascal GP102/GP104 run FP16 arithmetic at 1/64 rate: force the integer
        # MMQ kernels instead of FP16 cuBLAS. It is also the lower-VRAM path.
        if any(g.compute_cap == "6.1" for g in system.gpus):
            flags.append("-DGGML_CUDA_FORCE_MMQ=ON")
        if any(is_legacy(g.compute_cap) for g in system.gpus):
            flags.append("-DGGML_CUDA_FA_ALL_QUANTS=ON")

    flags.append("-DGGML_NATIVE=ON" if system.cpu.has("avx2") or not system.cpu.has("avx") else "-DGGML_NATIVE=ON")
    jobs = max(1, min(system.cpu.cores, 8))
    return [
        "git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp",
        "cmake -B build " + " ".join(flags),
        f"cmake --build build --config Release -j {jobs}",
    ]


@dataclass
class RunPlan:
    ctx: int
    ctk: str
    ctv: str
    n_gpu_layers: str
    budget: mem.Budget
    notes: list[str]
    threads: int | None = None
    decode_tps: float | None = None

    def command(self, model_arg: str) -> str:
        """`model_arg` is a ready flag: `-m <path>` or `-hf <repo>:<quant>`."""
        parts = [
            "llama-server",
            model_arg,
            f"-c {self.ctx}",
            f"-ngl {self.n_gpu_layers}",
            "-fa on",
        ]
        if self.ctk != "f16" or self.ctv != "f16":
            parts.append(f"-ctk {self.ctk} -ctv {self.ctv}")
        if self.threads:
            parts.append(f"-t {self.threads}")
        parts.append("--host 127.0.0.1 --port 8080")
        return " ".join(parts)


def plan_run(model: GgufModel, system: System, ctx: int | None = None,
             ubatch: int = 512) -> RunPlan:
    """Pick KV cache types and context so the model fits in the detected VRAM."""
    vram = system.total_vram_bytes
    gpu_count = max(len(system.gpus), 1)
    backend = system.backend
    notes: list[str] = []

    candidates = [("f16", "f16"), ("q8_0", "q8_0"), ("q8_0", "q4_0")]
    chosen = candidates[0]
    target_ctx = ctx or min(model.n_ctx_train or 8192, 32768)

    for ctk, ctv in candidates:
        fits_ctx = mem.max_context(model, vram, ctk, ctv, ubatch, gpu_count, backend=backend)
        if fits_ctx >= target_ctx:
            chosen = (ctk, ctv)
            break
        chosen = (ctk, ctv)
    ctk, ctv = chosen

    best_ctx = mem.max_context(model, vram, ctk, ctv, ubatch, gpu_count, backend=backend)
    final_ctx = ctx or min(target_ctx, best_ctx) or 2048
    budget = mem.plan(model, vram, final_ctx, ctk, ctv, ubatch, gpu_count, backend)

    if budget.fits:
        n_gpu_layers = "all"
    else:
        layers = mem.layers_that_fit(model, vram, final_ctx, ctk, ctv, ubatch,
                                     gpu_count, backend)
        n_gpu_layers = str(layers)
        # Report the budget for the plan actually being recommended, not for a
        # full offload that was already ruled out.
        budget = mem.plan(model, vram, final_ctx, ctk, ctv, ubatch, gpu_count,
                          backend, gpu_layers=layers)
        notes.append(
            f"Partial offload: {layers} of {model.n_layer} layers on the GPU, the rest "
            f"on the CPU ({mem.human(budget.weights_cpu)} of weights). Expect a large "
            "speed drop -- the bandwidth figure below no longer applies."
        )
        if best_ctx >= 1024 and ctx:
            notes.append(
                f"Context {best_ctx} would fit entirely on the GPU. Full offload at a "
                "shorter context is usually much faster than spilling layers to the CPU."
            )
        if model.is_moe:
            notes.append("MoE model: try -cmoe / -ncmoe N to keep expert weights on the CPU instead of whole layers.")

    if ctk in mem.QUANTIZED_CACHE or ctv in mem.QUANTIZED_CACHE:
        notes.append("Quantized KV cache requires Flash Attention, hence -fa on.")
    if ctx and ctx > best_ctx and budget.fits is False:
        notes.append(f"Requested context {ctx} is above the estimated maximum of {best_ctx}.")
    if model.sliding_window:
        notes.append(
            f"Model uses sliding-window attention (window {model.sliding_window}); real KV usage "
            "may be lower than this estimate."
        )
    if model.n_ctx_train and final_ctx > model.n_ctx_train:
        notes.append(f"Context exceeds the trained context length ({model.n_ctx_train}).")

    threads = None
    cpu = system.cpu
    if cpu.perf_cores and cpu.perf_cores < cpu.cores:
        threads = cpu.perf_cores

    bandwidth = system.gpus[0].bandwidth_gbs if system.gpus else None
    decode_tps = mem.decode_ceiling_tps(model.weights_bytes, bandwidth)
    if decode_tps and not budget.partial:
        if model.is_moe:
            notes.append(
                "MoE model: only a fraction of the weights is read per token, so real speed "
                "will be well above the bandwidth ceiling shown."
            )
        elif decode_tps < 5:
            notes.append(
                f"Memory bandwidth caps this at about {decode_tps:.1f} tok/s even fully "
                "offloaded. A smaller model or a heavier quant will feel much better."
            )

    return RunPlan(ctx=final_ctx, ctk=ctk, ctv=ctv, n_gpu_layers=n_gpu_layers,
                   budget=budget, notes=notes, threads=threads, decode_tps=decode_tps)
