"""Tests for the bench command."""

from __future__ import annotations

import json
import os
import stat
import struct
from pathlib import Path

import pytest

from oldiron import advisor, bench, memory, render
from oldiron.bench import BenchError, Run, build_command, find_binary, parse_output
from oldiron.gguf_meta import read_gguf
from oldiron.hardware import Cpu, Gpu, System

# A trimmed but faithful llama-bench `-o json` payload: one pp row, one tg row.
LLAMA_BENCH_JSON = """
[
  {
    "build_commit": "8cf427ff",
    "build_number": 6531,
    "cuda": true,
    "gpu_info": "Tesla P100-PCIE-16GB",
    "model_filename": "Qwen3.8-27B-UD-Q3_K_XL.gguf",
    "model_type": "qwen35 27B Q3_K_XL",
    "model_size": 13141400000,
    "n_threads": 8,
    "type_k": "q8_0",
    "type_v": "q4_0",
    "n_gpu_layers": 61,
    "flash_attn": true,
    "n_prompt": 512,
    "n_gen": 0,
    "avg_ns": 1100000000,
    "stddev_ns": 12000000,
    "avg_ts": 465.13,
    "stddev_ts": 5.02
  },
  {
    "build_commit": "8cf427ff",
    "gpu_info": "Tesla P100-PCIE-16GB",
    "n_prompt": 0,
    "n_gen": 128,
    "avg_ts": 11.42,
    "stddev_ts": 0.18
  }
]
"""


@pytest.fixture()
def model(tmp_path: Path):
    def s(text: str) -> bytes:
        raw = text.encode()
        return struct.pack("<Q", len(raw)) + raw

    kv = [
        ("general.architecture", 8, "qwen35"),
        ("general.name", 8, "Qwen3.8 27B"),
        ("qwen35.block_count", 4, 65),
        ("qwen35.embedding_length", 4, 5120),
        ("qwen35.context_length", 4, 262144),
        ("qwen35.attention.head_count", 4, 40),
        ("qwen35.attention.head_count_kv", 4, 8),
        ("qwen35.attention.key_length", 4, 128),
        ("qwen35.vocab_size", 4, 151936),
    ]
    body = b""
    for key, vtype, value in kv:
        body += s(key) + struct.pack("<I", vtype)
        body += s(value) if vtype == 8 else struct.pack("<I", value)
    head = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", len(kv))
    path = tmp_path / "Qwen3.8-27B-UD-Q3_K_XL.gguf"
    path.write_bytes(head + body + b"\x00" * 4096)
    parsed = read_gguf(path)
    # The real quant weighs 12.24 GiB; without that the bandwidth ceiling, and
    # every efficiency figure derived from it, is meaningless.
    parsed.weights_bytes = 13_141_400_000
    return parsed


def _p100() -> System:
    cpu = Cpu(model="Intel(R) Xeon(R) CPU E3-1230 V2", cores=8,
              flags={"avx", "sse4_2"}, isa="x86_64")
    gpu = Gpu(index=0, name="Tesla P100-PCIE-16GB", vram_mib=16384, compute_cap="6.0",
              arch="Pascal", bandwidth_gbs=732.0)
    return System(os="Linux 7.0", cpu=cpu, ram_bytes=23 * memory.GIB, gpus=[gpu],
                  driver_version="580.178.04", cuda_version="12.4")


# --- parsing ---------------------------------------------------------------

def test_parse_output_reads_both_tests():
    runs, commit, gpu = parse_output(LLAMA_BENCH_JSON)
    assert commit == "8cf427ff"
    assert gpu == "Tesla P100-PCIE-16GB"
    kinds = {r.kind: r for r in runs}
    assert kinds["pp"].avg_ts == 465.13
    assert kinds["tg"].avg_ts == 11.42
    assert kinds["tg"].stddev_ts == 0.18


def test_parse_output_tolerates_leading_log_lines():
    noisy = "ggml_cuda_init: found 1 device\nload_backend: loaded CUDA\n" + LLAMA_BENCH_JSON
    runs, _, _ = parse_output(noisy)
    assert len(runs) == 2


def test_parse_output_rejects_garbage():
    with pytest.raises(BenchError):
        parse_output("nothing here")
    with pytest.raises(BenchError):
        parse_output("[]")
    with pytest.raises(BenchError):
        parse_output('[{"n_prompt": 512}]')  # no avg_ts


# --- command building ------------------------------------------------------

def test_build_command_mirrors_the_plan(model):
    system = _p100()
    plan = advisor.plan_run(model, system, ctx=32768)
    cmd = build_command(Path("/usr/bin/llama-bench"), model, plan, 512, 128, 3)
    assert cmd[:3] == ["/usr/bin/llama-bench", "-m", str(model.path)]
    # the benchmark must use the settings fit recommends, not llama-bench defaults
    assert cmd[cmd.index("-ngl") + 1] == plan.n_gpu_layers
    assert cmd[cmd.index("-ctk") + 1] == plan.ctk
    assert cmd[cmd.index("-ctv") + 1] == plan.ctv
    assert cmd[cmd.index("-o") + 1] == "json"
    assert "-fa" in cmd


def test_build_command_translates_all_to_a_layer_count(model):
    system = _p100()
    plan = advisor.plan_run(model, system, ctx=2048)
    plan.n_gpu_layers = "all"
    cmd = build_command(Path("llama-bench"), model, plan)
    assert cmd[cmd.index("-ngl") + 1] == "999"


def test_build_command_passes_threads_when_recommended(model):
    system = _p100()
    plan = advisor.plan_run(model, system, ctx=2048)
    plan.threads = 4
    cmd = build_command(Path("llama-bench"), model, plan)
    assert cmd[cmd.index("-t") + 1] == "4"


def test_build_command_refuses_a_remote_model(model):
    plan = advisor.plan_run(model, _p100(), ctx=2048)
    model.remote_ref = "unsloth/Qwen3-4B-GGUF:Q4_K_M"
    with pytest.raises(BenchError) as excinfo:
        build_command(Path("llama-bench"), model, plan)
    assert "download it first" in str(excinfo.value)


# --- binary discovery ------------------------------------------------------

def test_find_binary_prefers_explicit_path(tmp_path):
    fake = tmp_path / "llama-bench"
    fake.write_text("#!/bin/sh\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    assert find_binary(str(fake)) == fake


def test_find_binary_rejects_a_bad_explicit_path(tmp_path):
    with pytest.raises(BenchError):
        find_binary(str(tmp_path / "nope"))


def test_find_binary_uses_path(tmp_path, monkeypatch):
    fake = tmp_path / "llama-bench"
    fake.write_text("#!/bin/sh\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", str(tmp_path))
    assert find_binary() == fake


def test_find_binary_looks_beside_llama_server(tmp_path, monkeypatch):
    # A tree built for llama-server keeps llama-bench in the same directory.
    server = tmp_path / "llama-server"
    server.write_text("#!/bin/sh\n")
    server.chmod(server.stat().st_mode | stat.S_IEXEC)
    sibling = tmp_path / "llama-bench"
    sibling.write_text("#!/bin/sh\n")
    sibling.chmod(sibling.stat().st_mode | stat.S_IEXEC)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "llama-server").symlink_to(server)
    monkeypatch.setenv("PATH", str(bin_dir))
    monkeypatch.setattr(bench, "SEARCH_DIRS", ())
    assert find_binary() == sibling


def test_find_binary_explains_itself_when_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.setattr(bench, "SEARCH_DIRS", ())
    with pytest.raises(BenchError) as excinfo:
        find_binary()
    assert "-t llama-bench" in str(excinfo.value)


# --- measuring and reporting -----------------------------------------------

def test_measure_computes_efficiency_against_the_ceiling(model):
    system = _p100()
    plan = advisor.plan_run(model, system, ctx=4096)  # fits entirely
    assert not plan.budget.partial
    result = bench.measure(model, system, plan, Path("llama-bench"),
                           runner=lambda cmd: LLAMA_BENCH_JSON)
    assert result.generation.avg_ts == 11.42
    assert result.prompt.avg_ts == 465.13
    assert result.ceiling_tps == plan.decode_tps
    # a P100 delivering 11.4 of a 55-ish ceiling is about a fifth of it
    assert 0.1 < result.efficiency < 0.35


def test_efficiency_is_withheld_under_partial_offload(model):
    """Dividing by a full-offload ceiling would measure the host CPU instead."""
    system = _p100()
    plan = advisor.plan_run(model, system, ctx=32768)
    assert plan.budget.partial
    result = bench.measure(model, system, plan, Path("llama-bench"),
                           runner=lambda cmd: LLAMA_BENCH_JSON)
    assert result.partial_offload
    assert result.efficiency is None
    assert result.generation.avg_ts == 11.42  # raw timings are still reported


def test_measure_reports_a_failing_binary(model):
    plan = advisor.plan_run(model, _p100(), ctx=2048)

    def failing(cmd):
        raise BenchError("llama-bench failed: CUDA error")

    with pytest.raises(BenchError):
        bench.measure(model, _p100(), plan, Path("llama-bench"), runner=failing)


def test_submission_record_is_complete(model):
    system = _p100()
    plan = advisor.plan_run(model, system, ctx=4096)
    result = bench.measure(model, system, plan, Path("llama-bench"),
                           runner=lambda cmd: LLAMA_BENCH_JSON)
    record = bench.submission(model, system, plan, result)

    assert record["device"]["name"] == "Tesla P100-PCIE-16GB"
    assert record["device"]["bandwidth_gbs"] == 732.0
    assert record["host"]["simd"] == "avx"
    assert record["host"]["driver"] == "580.178.04"
    assert record["model"]["quant"] == "Q3_K_XL"
    assert record["settings"]["cache_type_k"] == plan.ctk
    assert record["settings"]["partial_offload"] is plan.budget.partial
    assert record["results"]["generation_tps"] == 11.42
    assert record["results"]["llama_cpp_commit"] == "8cf427ff"
    assert 0 < record["results"]["bandwidth_efficiency"] < 1
    json.dumps(record)  # must stay serialisable


def test_suggested_filename_is_a_safe_slug(model):
    name = bench.suggested_filename(model, _p100())
    assert name.endswith(".json")
    assert " " not in name and "/" not in name
    assert "tesla-p100" in name


def test_write_submission_creates_directories(tmp_path, model):
    system = _p100()
    plan = advisor.plan_run(model, system, ctx=2048)
    result = bench.measure(model, system, plan, Path("llama-bench"),
                           runner=lambda cmd: LLAMA_BENCH_JSON)
    target = tmp_path / "results" / "x.json"
    written = bench.write_submission(bench.submission(model, system, plan, result), target)
    assert written.exists()
    assert json.loads(written.read_text())["device"]["arch"] == "Pascal"


# --- end to end with a fake llama-bench ------------------------------------

def test_end_to_end_with_a_fake_binary(tmp_path, model, monkeypatch):
    fake = tmp_path / "llama-bench"
    fake.write_text("#!/bin/sh\ncat <<'EOF'\n" + LLAMA_BENCH_JSON + "\nEOF\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)

    system = _p100()
    plan = advisor.plan_run(model, system, ctx=32768)
    result = bench.measure(model, system, plan, fake)  # real subprocess
    assert result.generation.avg_ts == 11.42
    assert result.binary == str(fake)


def test_renderers_flag_a_partial_offload(model):
    system = _p100()
    plan = advisor.plan_run(model, system, ctx=32768)
    result = bench.measure(model, system, plan, Path("llama-bench"),
                           runner=lambda cmd: LLAMA_BENCH_JSON)

    lines: list[str] = []
    render.PlainRenderer(write=lines.append).bench(result)
    plain = "\n".join(lines)
    assert "Partial offload" in plain
    assert "% of ceiling" not in plain

    pytest.importorskip("rich")
    import io
    from rich.console import Console
    buffer = io.StringIO()
    render.RichRenderer(Console(file=buffer, width=100, no_color=True)).bench(result)
    text = buffer.getvalue()
    assert "partial offload" in text
    assert "% of the ceiling" not in text


def test_renderers_show_the_measurement(model):
    system = _p100()
    plan = advisor.plan_run(model, system, ctx=4096)
    result = bench.measure(model, system, plan, Path("llama-bench"),
                           runner=lambda cmd: LLAMA_BENCH_JSON)

    lines: list[str] = []
    render.PlainRenderer(write=lines.append).bench(result, "results/x.json")
    plain = "\n".join(lines)
    assert "11.42" in plain and "465.13" in plain
    assert "ceiling" in plain
    assert "results/x.json" in plain

    rich = pytest.importorskip("rich")
    import io
    from rich.console import Console
    buffer = io.StringIO()
    console = Console(file=buffer, width=100, no_color=True, legacy_windows=False)
    render.RichRenderer(console).bench(result, "results/x.json")
    text = buffer.getvalue()
    assert "11.42" in text
    assert "% of the ceiling" in text
