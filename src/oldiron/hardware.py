"""Detect GPUs, CPU instruction sets, RAM, driver and CUDA toolkit.

Pure standard library: this has to run on machines where compiling or even
installing a wheel is the problem we are diagnosing.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field

from .gpudb import ArchInfo, apple_chip_for, arch_for

NVIDIA_SMI_QUERY = "name,memory.total,compute_cap,driver_version"

# Metal exposes recommendedMaxWorkingSetSize, about 75% of unified memory, as the
# ceiling for GPU allocations. Raise it with: sudo sysctl iogpu.wired_limit_mb=N
APPLE_GPU_SHARE = 0.75

# macOS x86 sysctl spellings -> the /proc/cpuinfo flag names we use elsewhere.
_DARWIN_X86_FLAGS = {
    "avx1.0": "avx", "avx2": "avx2", "avx512f": "avx512f",
    "sse4.2": "sse4_2", "sse4.1": "sse4_1", "fma": "fma", "f16c": "f16c",
}

# Arm features that matter for llama.cpp CPU kernels, mapped to the sysctl keys
# that expose them. Apple follows Arm's own capitalization, so FEAT_DotProd is
# mixed case while FEAT_I8MM is not -- and sysctl keys are case-sensitive.
# Older macOS releases only had the armv8_* spellings, hence the fallbacks.
_ARM_FEATURES: dict[str, tuple[str, ...]] = {
    "FEAT_DotProd": ("hw.optional.arm.FEAT_DotProd", "hw.optional.armv8_2_dotprod"),
    "FEAT_I8MM": ("hw.optional.arm.FEAT_I8MM",),
    "FEAT_BF16": ("hw.optional.arm.FEAT_BF16",),
    "FEAT_FP16": ("hw.optional.arm.FEAT_FP16", "hw.optional.neon_fp16"),
    "FEAT_SME": ("hw.optional.arm.FEAT_SME",),
}

# Arm architecture implications: a CPU cannot have the later feature without the
# earlier one, so a missing sysctl key never turns into a bogus warning.
_ARM_IMPLIES: dict[str, tuple[str, ...]] = {
    "FEAT_I8MM": ("FEAT_DotProd",),
    "FEAT_BF16": ("FEAT_DotProd",),
    "FEAT_SME": ("FEAT_DotProd", "FEAT_BF16"),
}


@dataclass
class Gpu:
    index: int
    name: str
    vram_mib: int
    compute_cap: str | None
    arch: str | None = None
    unified: bool = False  # memory shared with the CPU (Apple Silicon)
    bandwidth_gbs: float | None = None
    cores: int | None = None

    @property
    def vram_bytes(self) -> int:
        return self.vram_mib * 1024 * 1024


@dataclass
class Cpu:
    model: str
    cores: int
    flags: set[str] = field(default_factory=set)
    isa: str = "x86_64"  # "x86_64" or "arm64"
    perf_cores: int | None = None  # Apple Silicon performance cores

    def has(self, flag: str) -> bool:
        """Case-insensitive: Arm spells features FEAT_DotProd, x86 uses lowercase."""
        wanted = flag.lower()
        return any(f.lower() == wanted for f in self.flags)

    @property
    def is_arm(self) -> bool:
        return self.isa == "arm64"

    @property
    def simd_level(self) -> str:
        if self.is_arm:
            named = [f.replace("FEAT_", "").lower() for f in _ARM_FEATURES if self.has(f)]
            return "neon" + (" + " + ", ".join(named) if named else "")
        if self.has("avx512f"):
            return "avx512"
        if self.has("avx2"):
            return "avx2"
        if self.has("avx"):
            return "avx"
        if self.has("sse4_2"):
            return "sse4.2"
        return "baseline"

    @property
    def threads_hint(self) -> int:
        """Threads worth giving llama.cpp: efficiency cores usually hurt."""
        return self.perf_cores or max(1, self.cores)


@dataclass
class System:
    os: str
    cpu: Cpu
    ram_bytes: int
    gpus: list[Gpu]
    driver_version: str | None = None
    cuda_version: str | None = None
    platform: str = "linux"  # "linux", "darwin", "windows"

    @property
    def total_vram_bytes(self) -> int:
        return sum(g.vram_bytes for g in self.gpus)

    @property
    def unified_memory(self) -> bool:
        return bool(self.gpus) and all(g.unified for g in self.gpus)

    @property
    def backend(self) -> str:
        if not self.gpus:
            return "cpu"
        return "metal" if self.unified_memory else "cuda"

    def to_dict(self) -> dict:
        data = asdict(self)
        data["cpu"]["flags"] = sorted(self.cpu.flags)
        data["backend"] = self.backend
        return data


def _run(cmd: list[str], timeout: int = 20) -> str | None:
    if not shutil.which(cmd[0]):
        return None
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout


def _sysctl(name: str) -> str | None:
    out = _run(["sysctl", "-n", name], timeout=5)
    return out.strip() if out and out.strip() else None


# --- NVIDIA ----------------------------------------------------------------

def parse_nvidia_smi(text: str) -> tuple[list[Gpu], str | None]:
    """Parse `nvidia-smi --query-gpu=... --format=csv,noheader` output."""
    gpus: list[Gpu] = []
    driver: str | None = None
    for i, line in enumerate(text.strip().splitlines()):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        name = parts[0]
        mem = re.sub(r"[^0-9]", "", parts[1]) or "0"
        cc = parts[2] if re.match(r"^\d+\.\d+$", parts[2]) else None
        if len(parts) > 3 and parts[3] and parts[3] != "[N/A]":
            driver = parts[3]
        info = arch_for(cc, name)
        gpus.append(
            Gpu(
                index=i,
                name=name,
                vram_mib=int(mem),
                compute_cap=cc or (info.cc if info else None),
                arch=info.name if info else None,
                bandwidth_gbs=info.bandwidth_gbs if info else None,
            )
        )
    return gpus, driver


def detect_gpus() -> tuple[list[Gpu], str | None]:
    out = _run(["nvidia-smi", f"--query-gpu={NVIDIA_SMI_QUERY}", "--format=csv,noheader"])
    if not out:
        return [], None
    return parse_nvidia_smi(out)


def parse_nvcc_version(text: str) -> str | None:
    match = re.search(r"release (\d+\.\d+)", text)
    return match.group(1) if match else None


def detect_cuda() -> str | None:
    out = _run(["nvcc", "--version"])
    return parse_nvcc_version(out) if out else None


# --- CPU -------------------------------------------------------------------

def parse_cpuinfo(text: str) -> Cpu:
    """Parse Linux /proc/cpuinfo."""
    model = ""
    flags: set[str] = set()
    cores = 0
    isa = "x86_64"
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key, value = key.strip().lower(), value.strip()
        if key == "model name" and not model:
            model = value
        elif key in ("flags", "features") and not flags:
            flags = set(value.split())
            if key == "features":  # aarch64 spelling
                isa = "arm64"
        elif key == "processor":
            cores += 1
    if "aarch64" in platform.machine().lower() or "arm64" in platform.machine().lower():
        isa = "arm64"
    return Cpu(
        model=model or platform.processor() or "unknown",
        cores=cores or (os.cpu_count() or 1),
        flags=flags,
        isa=isa,
    )


def implied_arm_features(flags: set[str]) -> set[str]:
    """Features guaranteed by the ones already detected."""
    extra: set[str] = set()
    for feature, implied in _ARM_IMPLIES.items():
        if feature in flags:
            extra.update(implied)
    return extra - flags


def parse_darwin_x86_features(text: str) -> set[str]:
    """Map `sysctl machdep.cpu.features` + leaf7_features onto cpuinfo names."""
    flags = set()
    for token in text.replace(",", " ").split():
        mapped = _DARWIN_X86_FLAGS.get(token.lower())
        if mapped:
            flags.add(mapped)
    return flags


def detect_cpu_darwin() -> Cpu:
    machine = platform.machine().lower()
    isa = "arm64" if machine in ("arm64", "aarch64") else "x86_64"
    model = _sysctl("machdep.cpu.brand_string") or platform.processor() or machine
    threads = int(_sysctl("hw.logicalcpu") or os.cpu_count() or 1)

    flags: set[str] = set()
    perf_cores: int | None = None
    if isa == "arm64":
        for feature, keys in _ARM_FEATURES.items():
            if any(_sysctl(key) == "1" for key in keys):
                flags.add(feature)
        flags |= implied_arm_features(flags)
        value = _sysctl("hw.perflevel0.physicalcpu")
        perf_cores = int(value) if value and value.isdigit() else None
    else:
        text = " ".join(filter(None, (
            _sysctl("machdep.cpu.features"), _sysctl("machdep.cpu.leaf7_features")
        )))
        flags = parse_darwin_x86_features(text)

    return Cpu(model=model, cores=threads, flags=flags, isa=isa, perf_cores=perf_cores)


def detect_cpu() -> Cpu:
    if platform.system() == "Darwin":
        return detect_cpu_darwin()
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as fh:
            return parse_cpuinfo(fh.read())
    except OSError:
        machine = platform.machine().lower()
        return Cpu(
            model=platform.processor() or machine,
            cores=os.cpu_count() or 1,
            isa="arm64" if machine in ("arm64", "aarch64") else "x86_64",
        )


# --- memory ----------------------------------------------------------------

def detect_ram() -> int:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        pass
    value = _sysctl("hw.memsize")
    return int(value) if value and value.isdigit() else 0


def parse_gpu_core_count(text: str) -> int | None:
    match = re.search(r"Total Number of Cores:\s*(\d+)", text)
    return int(match.group(1)) if match else None


def apple_gpu(cpu: Cpu, ram_bytes: int) -> Gpu | None:
    """Describe the integrated GPU on Apple Silicon as a unified-memory device."""
    if cpu.isa != "arm64" or not ram_bytes:
        return None
    chip = apple_chip_for(cpu.model)
    limit = _sysctl("iogpu.wired_limit_mb")
    if limit and limit.isdigit() and int(limit) > 0:
        vram_mib = int(limit)
    else:
        vram_mib = int(ram_bytes * APPLE_GPU_SHARE) // (1024 * 1024)
    cores = None
    profile = _run(["system_profiler", "SPDisplaysDataType"], timeout=15)
    if profile:
        cores = parse_gpu_core_count(profile)
    return Gpu(
        index=0,
        name=cpu.model or "Apple Silicon GPU",
        vram_mib=vram_mib,
        compute_cap=None,
        arch="Apple Silicon",
        unified=True,
        bandwidth_gbs=chip.bandwidth_gbs if chip else None,
        cores=cores,
    )


def detect() -> System:
    cpu = detect_cpu()
    ram = detect_ram()
    gpus, driver = detect_gpus()
    system_name = platform.system()

    if not gpus and system_name == "Darwin":
        integrated = apple_gpu(cpu, ram)
        if integrated:
            gpus = [integrated]

    return System(
        os=f"{system_name} {platform.release()}",
        cpu=cpu,
        ram_bytes=ram,
        gpus=gpus,
        driver_version=driver,
        cuda_version=detect_cuda(),
        platform=system_name.lower(),
    )


def arch_of(gpu: Gpu) -> ArchInfo | None:
    return arch_for(gpu.compute_cap, gpu.name)
