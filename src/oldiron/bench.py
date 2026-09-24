"""Measure real throughput with llama-bench and compare it to the ceiling.

`fit` predicts; `bench` checks. The gap between the memory-bandwidth ceiling and
what a card actually delivers is the number nobody publishes for old hardware,
so a run here also produces a record that can be submitted to a shared table.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import __version__
from .advisor import RunPlan
from .gguf_meta import GgufModel
from .hardware import System
from .memory import GIB

BINARY_NAMES = ("llama-bench", "llama-bench.exe")

# Where a self-built llama.cpp usually lands, relative to cwd or home.
SEARCH_DIRS = (
    Path("build/bin"),
    Path("llama.cpp/build/bin"),
    Path.home() / "llama.cpp" / "build" / "bin",
    Path.home() / "src" / "llama.cpp" / "build" / "bin",
    Path("/usr/local/bin"),
    Path("/opt/llama.cpp/bin"),
)


class BenchError(Exception):
    pass


@dataclass
class Run:
    """One llama-bench measurement."""

    n_prompt: int
    n_gen: int
    avg_ts: float
    stddev_ts: float = 0.0

    @property
    def kind(self) -> str:
        return "pp" if self.n_prompt else "tg"


@dataclass
class BenchResult:
    runs: list[Run]
    ceiling_tps: float | None
    build_commit: str = ""
    gpu_info: str = ""
    binary: str = ""
    command: list[str] = field(default_factory=list)
    partial_offload: bool = False

    def _first(self, kind: str) -> Run | None:
        return next((r for r in self.runs if r.kind == kind), None)

    @property
    def prompt(self) -> Run | None:
        return self._first("pp")

    @property
    def generation(self) -> Run | None:
        return self._first("tg")

    @property
    def efficiency(self) -> float | None:
        """Measured generation speed as a fraction of the bandwidth ceiling.

        Undefined under a partial offload: the ceiling assumes every weight is in
        GPU memory, so dividing by it would measure the host CPU, not the card,
        and the resulting figure is not comparable with anyone else's.
        """
        gen = self.generation
        if not gen or not self.ceiling_tps or self.partial_offload:
            return None
        return gen.avg_ts / self.ceiling_tps


def find_binary(explicit: str | None = None) -> Path:
    """Locate llama-bench, preferring an explicit path, then PATH, then the usual spots."""
    if explicit:
        path = Path(explicit).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return path
        raise BenchError(f"{explicit} is not an executable file")

    for name in BINARY_NAMES:
        found = shutil.which(name)
        if found:
            return Path(found)

    # llama-bench is built from the same tree as the tools people actually run,
    # so if one of those is on PATH, look beside it.
    for sibling in ("llama-server", "llama-cli"):
        found = shutil.which(sibling)
        if not found:
            continue
        # Follow symlinks too: a binary is often linked into /usr/local/bin from
        # the build directory, and that directory is where llama-bench lives.
        for parent in {Path(found).parent, Path(found).resolve().parent}:
            for name in BINARY_NAMES:
                candidate = parent / name
                if candidate.is_file() and os.access(candidate, os.X_OK):
                    return candidate

    for directory in SEARCH_DIRS:
        for name in BINARY_NAMES:
            candidate = directory / name
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate

    raise BenchError(
        "llama-bench not found. It is a separate build target, so a tree built only for "
        "llama-server will not have it: run `cmake --build build --config Release "
        "-t llama-bench` in your llama.cpp checkout. Then pass "
        "--llama-bench /path/to/llama-bench, or add it to PATH."
    )


def build_command(binary: Path, model: GgufModel, plan: RunPlan,
                  prompt_tokens: int = 512, gen_tokens: int = 128,
                  repetitions: int = 3) -> list[str]:
    """Bench exactly the configuration `fit` recommends, not llama-bench's defaults."""
    if model.remote_ref:
        raise BenchError(
            "benchmarking needs the model on disk; download it first, then point "
            "oldiron bench at the .gguf file"
        )
    cmd = [
        str(binary),
        "-m", str(model.path),
        "-p", str(prompt_tokens),
        "-n", str(gen_tokens),
        "-r", str(repetitions),
        "-ngl", "999" if plan.n_gpu_layers == "all" else str(plan.n_gpu_layers),
        "-fa", "on",
        "-ctk", plan.ctk,
        "-ctv", plan.ctv,
        "-o", "json",
    ]
    if plan.threads:
        cmd += ["-t", str(plan.threads)]
    return cmd


def parse_output(text: str) -> tuple[list[Run], str, str]:
    """Parse llama-bench `-o json`: a list of objects, one per test."""
    text = text.strip()
    start = text.find("[")
    if start == -1:
        raise BenchError("llama-bench produced no JSON; run it by hand to see why")
    try:
        payload = json.loads(text[start:])
    except json.JSONDecodeError as exc:
        raise BenchError(f"cannot parse llama-bench output: {exc}") from exc
    if not isinstance(payload, list) or not payload:
        raise BenchError("llama-bench returned no measurements")

    runs: list[Run] = []
    commit = gpu = ""
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        commit = commit or str(entry.get("build_commit", ""))
        gpu = gpu or str(entry.get("gpu_info", ""))
        try:
            runs.append(Run(
                n_prompt=int(entry.get("n_prompt", 0)),
                n_gen=int(entry.get("n_gen", 0)),
                avg_ts=float(entry["avg_ts"]),
                stddev_ts=float(entry.get("stddev_ts", 0.0) or 0.0),
            ))
        except (KeyError, TypeError, ValueError):
            continue
    if not runs:
        raise BenchError("llama-bench output contained no usable timings")
    return runs, commit, gpu


def run_bench(command: list[str], timeout: int = 1800) -> str:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as exc:
        raise BenchError(f"cannot execute {command[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise BenchError(f"llama-bench did not finish within {timeout}s") from exc
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout or "").strip().splitlines()
        detail = tail[-1] if tail else f"exit code {completed.returncode}"
        raise BenchError(f"llama-bench failed: {detail}")
    return completed.stdout


def measure(model: GgufModel, system: System, plan: RunPlan,
            binary: Path, prompt_tokens: int = 512, gen_tokens: int = 128,
            repetitions: int = 3, runner=run_bench) -> BenchResult:
    command = build_command(binary, model, plan, prompt_tokens, gen_tokens, repetitions)
    runs, commit, gpu = parse_output(runner(command))
    return BenchResult(
        runs=runs,
        ceiling_tps=plan.decode_tps,
        build_commit=commit,
        gpu_info=gpu,
        binary=str(binary),
        command=command,
        partial_offload=plan.budget.partial,
    )


def submission(model: GgufModel, system: System, plan: RunPlan,
               result: BenchResult) -> dict:
    """A record suitable for the shared results table."""
    gpu = system.gpus[0] if system.gpus else None
    prompt, generation = result.prompt, result.generation
    return {
        "oldiron_version": __version__,
        "device": {
            "name": gpu.name if gpu else "cpu only",
            "arch": gpu.arch if gpu else None,
            "compute_capability": gpu.compute_cap if gpu else None,
            "memory_gib": round(gpu.vram_bytes / GIB, 2) if gpu else None,
            "bandwidth_gbs": gpu.bandwidth_gbs if gpu else None,
            "unified_memory": bool(gpu and gpu.unified),
            "count": len(system.gpus),
        },
        "host": {
            "os": system.os,
            "platform": platform.machine(),
            "cpu": system.cpu.model.strip(),
            "simd": system.cpu.simd_level,
            "ram_gib": round(system.ram_bytes / GIB, 2) if system.ram_bytes else None,
            "driver": system.driver_version,
            "cuda": system.cuda_version,
            "backend": system.backend,
        },
        "model": {
            "name": model.name,
            "arch": model.arch,
            "quant": model.file_type,
            "layers": model.n_layer,
            "weights_gib": round(model.weights_bytes / GIB, 2),
            "moe": model.is_moe,
        },
        "settings": {
            "n_gpu_layers": plan.n_gpu_layers,
            "partial_offload": plan.budget.partial,
            "cache_type_k": plan.ctk,
            "cache_type_v": plan.ctv,
            "threads": plan.threads,
            "flash_attn": True,
        },
        "results": {
            "prompt_tps": round(prompt.avg_ts, 2) if prompt else None,
            "generation_tps": round(generation.avg_ts, 2) if generation else None,
            "generation_stddev": round(generation.stddev_ts, 2) if generation else None,
            "ceiling_tps": round(result.ceiling_tps, 2) if result.ceiling_tps else None,
            "bandwidth_efficiency": (round(result.efficiency, 3)
                                     if result.efficiency else None),
            "llama_cpp_commit": result.build_commit,
        },
    }


def suggested_filename(model: GgufModel, system: System) -> str:
    gpu = system.gpus[0].name if system.gpus else "cpu"
    parts = [gpu, model.name, model.file_type]
    slug = "-".join("".join(c if c.isalnum() else "-" for c in p) for p in parts)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return f"{slug.strip('-').lower()}.json"


def write_submission(data: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path
