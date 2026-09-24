"""Knowledge base about NVIDIA GPU architectures, with a bias towards old ones."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ArchInfo:
    name: str  # architecture name
    cc: str  # compute capability, e.g. "6.1"
    max_cuda: str | None  # last CUDA major.minor that supports it, None = current
    max_driver: str | None  # last driver branch, None = current
    fp16_rate: str  # "fast", "slow", "none" -- FP16 arithmetic throughput
    int8_tensor_cores: bool
    notes: tuple[str, ...] = ()
    bandwidth_gbs: float | None = None  # memory bandwidth, the decode-speed ceiling


@dataclass(frozen=True)
class AppleChip:
    name: str
    bandwidth_gbs: float


# Unified memory bandwidth per Apple Silicon family. Decode speed for a local
# model is bandwidth-bound, so this is the number that predicts tokens/s.
APPLE_CHIPS: tuple[AppleChip, ...] = (
    AppleChip("M1", 68.0), AppleChip("M1 Pro", 200.0),
    AppleChip("M1 Max", 400.0), AppleChip("M1 Ultra", 800.0),
    AppleChip("M2", 100.0), AppleChip("M2 Pro", 200.0),
    AppleChip("M2 Max", 400.0), AppleChip("M2 Ultra", 800.0),
    AppleChip("M3", 100.0), AppleChip("M3 Pro", 150.0),
    AppleChip("M3 Max", 400.0), AppleChip("M3 Ultra", 800.0),
    AppleChip("M4", 120.0), AppleChip("M4 Pro", 273.0), AppleChip("M4 Max", 546.0),
)


def apple_chip_for(model: str) -> AppleChip | None:
    """Longest name first, so 'Apple M1 Max' does not match plain 'M1'."""
    lowered = (model or "").lower()
    for chip in sorted(APPLE_CHIPS, key=lambda c: -len(c.name)):
        if chip.name.lower() in lowered:
            return chip
    return None


# Compute capability -> architecture facts.
# Sources: NVIDIA CUDA toolkit release notes (CUDA 13.0 removed Maxwell/Pascal/Volta)
# and the NVIDIA UNIX driver deprecation schedule (r580 is the last branch for them).
ARCHS: dict[str, ArchInfo] = {
    "3.5": ArchInfo("Kepler", "3.5", "11.8", "470", "none", False,
                    ("Kepler needs CUDA 11.x and driver branch 470.",), 288.0),
    "3.7": ArchInfo("Kepler", "3.7", "11.8", "470", "none", False,
                    ("Kepler needs CUDA 11.x and driver branch 470.",), 240.0),
    "5.0": ArchInfo("Maxwell", "5.0", "12.9", "580", "none", False, (), 224.0),
    "5.2": ArchInfo("Maxwell", "5.2", "12.9", "580", "none", False, (), 288.0),
    "6.0": ArchInfo("Pascal", "6.0", "12.9", "580", "fast", False,
                    ("GP100 has full-rate FP16 (2x FP32) and HBM2 -- unusual for Pascal.",),
                    732.0),
    "6.1": ArchInfo("Pascal", "6.1", "12.9", "580", "slow", False,
                    ("GP102/104 run FP16 arithmetic at 1/64 rate; keep math in FP32/INT8.",),
                    347.0),
    "7.0": ArchInfo("Volta", "7.0", "12.9", "580", "fast", False, (), 900.0),
    "7.5": ArchInfo("Turing", "7.5", None, None, "fast", True, (), 616.0),
    "8.0": ArchInfo("Ampere", "8.0", None, None, "fast", True, (), 1555.0),
    "8.6": ArchInfo("Ampere", "8.6", None, None, "fast", True, (), 936.0),
    "8.9": ArchInfo("Ada Lovelace", "8.9", None, None, "fast", True, (), 1008.0),
    "9.0": ArchInfo("Hopper", "9.0", None, None, "fast", True, (), 3350.0),
    "10.0": ArchInfo("Blackwell", "10.0", None, None, "fast", True, (), 8000.0),
    "12.0": ArchInfo("Blackwell", "12.0", None, None, "fast", True, (), 1792.0),
}

# Fallback name lookup when the compute capability is unknown to us.
NAME_TO_CC = {
    "tesla k80": "3.7",
    "tesla m40": "5.2",
    "tesla m60": "5.2",
    "tesla p100": "6.0",
    "tesla p40": "6.1",
    "tesla p4": "6.1",
    "quadro p6000": "6.1",
}

LEGACY_CC = {"3.5", "3.7", "5.0", "5.2", "6.0", "6.1", "7.0"}


def arch_for(cc: str | None, name: str = "") -> ArchInfo | None:
    if cc and cc in ARCHS:
        return ARCHS[cc]
    lowered = name.lower()
    for key, mapped in NAME_TO_CC.items():
        if key in lowered:
            return ARCHS[mapped]
    return None


def is_legacy(cc: str | None) -> bool:
    return bool(cc) and cc in LEGACY_CC
