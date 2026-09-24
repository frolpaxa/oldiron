"""Tests for the two renderers."""

import io
import struct
from pathlib import Path

import pytest

from oldiron import advisor, memory, render
from oldiron.gguf_meta import read_gguf
from oldiron.hardware import Cpu, Gpu, System
from oldiron.render import PlainRenderer, get_renderer

rich = pytest.importorskip("rich")
from rich.console import Console  # noqa: E402


def _system(fits_ram_gib: int = 32) -> System:
    cpu = Cpu(model="Intel(R) Xeon(R) CPU E3-1230 V2", cores=8,
              flags={"avx", "sse4_2"}, isa="x86_64")
    gpu = Gpu(index=0, name="Tesla P100-PCIE-16GB", vram_mib=16384,
              compute_cap="6.0", arch="Pascal", bandwidth_gbs=732.0)
    return System(os="Linux 6.1", cpu=cpu, ram_bytes=fits_ram_gib * memory.GIB,
                  gpus=[gpu], driver_version="580.65.06", cuda_version="12.4")


@pytest.fixture()
def model(tmp_path: Path):
    def s(text: str) -> bytes:
        raw = text.encode()
        return struct.pack("<Q", len(raw)) + raw

    kv = [
        ("general.architecture", 8, "llama"),
        ("general.name", 8, "Bracket [test] Model"),  # markup must not be parsed
        ("llama.block_count", 4, 32),
        ("llama.embedding_length", 4, 4096),
        ("llama.context_length", 4, 8192),
        ("llama.attention.head_count", 4, 32),
        ("llama.attention.head_count_kv", 4, 8),
        ("llama.attention.key_length", 4, 128),
        ("llama.vocab_size", 4, 32000),
    ]
    body = b""
    for key, vtype, value in kv:
        body += s(key) + struct.pack("<I", vtype)
        body += s(value) if vtype == 8 else struct.pack("<I", value)
    head = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", len(kv))
    path = tmp_path / "Bracket-Q4_K_M.gguf"
    path.write_bytes(head + body + b"\x00" * 4096)
    return read_gguf(path)


def _rich_output(callback) -> str:
    buffer = io.StringIO()
    console = Console(file=buffer, width=100, no_color=True, legacy_windows=False)
    callback(render.RichRenderer(console))
    return buffer.getvalue()


def _plain_output(callback) -> str:
    lines: list[str] = []
    callback(PlainRenderer(write=lines.append))
    return "\n".join(lines)


# --- selection -------------------------------------------------------------

def test_get_renderer_honours_plain_flag():
    assert isinstance(get_renderer(plain=True), PlainRenderer)
    assert type(get_renderer(plain=False)).__name__ == "RichRenderer"


def test_falls_back_to_plain_without_rich(monkeypatch):
    monkeypatch.setattr(render, "HAS_RICH", False)
    assert isinstance(get_renderer(plain=False), PlainRenderer)


# --- content parity --------------------------------------------------------

def test_both_renderers_report_the_same_hardware():
    system = _system()
    for text in (_plain_output(lambda ui: ui.system(system)),
                 _rich_output(lambda ui: ui.system(system))):
        assert "Tesla P100-PCIE-16GB" in text
        assert "16.00 GiB" in text
        assert "580.65.06" in text
        assert "12.4" in text


def test_both_renderers_show_findings_and_fixes():
    findings = advisor.check_system(_system())
    for text in (_plain_output(lambda ui: ui.findings(findings)),
                 _rich_output(lambda ui: ui.findings(findings))):
        assert "Legacy GPU detected" in text
        assert "CUDA 13.0" in text
        assert "SIGILL" in text  # the AVX2 warning survives both layouts


def test_both_renderers_show_the_budget(model):
    system = _system()
    plan = advisor.plan_run(model, system, ctx=4096)
    for text in (_plain_output(lambda ui: ui.plan(plan, system)),
                 _rich_output(lambda ui: ui.plan(plan, system))):
        assert "weights" in text
        assert "KV cache" in text
        assert "4096" in text or "4,096" in text


def test_rich_marks_a_partial_offload(model, monkeypatch):
    system = _system()
    monkeypatch.setattr(model, "weights_bytes", 40 * memory.GIB, raising=False)
    plan = advisor.plan_run(model, system, ctx=4096)
    text = _rich_output(lambda ui: ui.plan(plan, system))
    assert "partial offload" in text.lower()
    assert "weights on GPU" in text
    assert "on CPU" in text
    # the bandwidth ceiling must not be presented as achievable
    assert "does not apply" in text


def test_plain_renderer_reports_the_cpu_remainder(model, monkeypatch):
    system = _system()
    monkeypatch.setattr(model, "weights_bytes", 40 * memory.GIB, raising=False)
    plan = advisor.plan_run(model, system, ctx=4096)
    text = _plain_output(lambda ui: ui.plan(plan, system))
    assert "weights on GPU" in text
    assert "on CPU" in text
    assert "does not apply" in text


def test_speed_line_is_shown_when_it_fits(model):
    system = _system()
    plan = advisor.plan_run(model, system, ctx=4096)
    assert plan.decode_tps
    for text in (_plain_output(lambda ui: ui.plan(plan, system)),
                 _rich_output(lambda ui: ui.plan(plan, system))):
        assert "tok/s" in text


# --- rich specifics --------------------------------------------------------

def test_model_name_with_brackets_is_not_parsed_as_markup(model):
    text = _rich_output(lambda ui: ui.model(model))
    assert "Bracket [test] Model" in text


def test_budget_bar_marks_overflow(model):
    system = _system()
    renderer = render.RichRenderer(Console(file=io.StringIO(), width=100))
    # A context so large that the KV cache alone cannot fit, with or without
    # offloading layers -- this is the genuinely impossible case.
    doomed = advisor.plan_run(model, system, ctx=2_000_000)
    assert not doomed.budget.fits
    assert "▓" in renderer._budget_bar(doomed.budget).plain

    ok_plan = advisor.plan_run(model, system, ctx=2048)
    assert ok_plan.budget.fits
    assert "░" in renderer._budget_bar(ok_plan.budget).plain


def test_context_table_heading_is_outside_the_table():
    text = _rich_output(lambda ui: ui.context_table([("f16/f16", 8192), ("q4_0/q4_0", 0)]))
    lines = [line for line in text.splitlines() if line.strip()]
    assert lines[0].strip() == "max context, fully offloaded"
    assert "8,192" in text
    assert "does not fit" in text


def test_error_is_visible_in_both_renderers():
    assert "boom" in _plain_output(lambda ui: ui.error("boom"))
    assert "boom" in _rich_output(lambda ui: ui.error("boom"))
