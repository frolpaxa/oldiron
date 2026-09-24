import struct
from pathlib import Path

import pytest

from oldiron import advisor, memory
from oldiron.gguf_meta import GgufError, read_gguf
from oldiron.gpudb import apple_chip_for
from oldiron.hardware import (
    Cpu,
    Gpu,
    System,
    implied_arm_features,
    parse_cpuinfo,
    parse_darwin_x86_features,
    parse_gpu_core_count,
    parse_nvcc_version,
    parse_nvidia_smi,
)

P100_SMI = "Tesla P100-PCIE-16GB, 16384 MiB, 6.0, 580.65.06\n"
P40_SMI = "Tesla P40, 24576 MiB, 6.1, 590.44.01\n"
MIXED_SMI = P100_SMI + "NVIDIA GeForce RTX 3090, 24576 MiB, 8.6, 580.65.06\n"

IVY_BRIDGE_CPUINFO = """processor\t: 0
model name\t: Intel(R) Xeon(R) CPU E3-1230 V2 @ 3.30GHz
flags\t\t: fpu vme de pse tsc msr sse sse2 ssse3 sse4_1 sse4_2 avx f16c rdrand
processor\t: 1
model name\t: Intel(R) Xeon(R) CPU E3-1230 V2 @ 3.30GHz
flags\t\t: fpu vme de pse tsc msr sse sse2 ssse3 sse4_1 sse4_2 avx f16c rdrand
"""

MODERN_CPUINFO = """processor\t: 0
model name\t: AMD Ryzen 9 7950X
flags\t\t: fpu sse sse2 avx avx2 fma f16c avx512f
"""


# --- hardware parsing ------------------------------------------------------

def test_parse_nvidia_smi_reads_vram_and_arch():
    gpus, driver = parse_nvidia_smi(P100_SMI)
    assert len(gpus) == 1
    assert gpus[0].name == "Tesla P100-PCIE-16GB"
    assert gpus[0].vram_mib == 16384
    assert gpus[0].compute_cap == "6.0"
    assert gpus[0].arch == "Pascal"
    assert driver == "580.65.06"


def test_parse_cpuinfo_detects_missing_avx2():
    cpu = parse_cpuinfo(IVY_BRIDGE_CPUINFO)
    assert cpu.cores == 2
    assert cpu.has("avx")
    assert not cpu.has("avx2")
    assert cpu.simd_level == "avx"


def test_parse_nvcc_version():
    assert parse_nvcc_version("Cuda compilation tools, release 12.4, V12.4.131") == "12.4"
    assert parse_nvcc_version("nonsense") is None


# --- findings --------------------------------------------------------------

def _system(smi: str, cpuinfo: str, cuda: str | None, ram_gib: int = 32) -> System:
    gpus, driver = parse_nvidia_smi(smi)
    return System(
        os="Linux 6.1", cpu=parse_cpuinfo(cpuinfo), ram_bytes=ram_gib * memory.GIB,
        gpus=gpus, driver_version=driver, cuda_version=cuda,
    )


def _titles(findings):
    return [f.title for f in findings]


def test_pascal_with_cuda12_is_fine_but_flagged_as_legacy():
    findings = advisor.check_system(_system(P100_SMI, IVY_BRIDGE_CPUINFO, "12.4"))
    assert "Legacy GPU detected" in _titles(findings)
    assert not [f for f in findings if f.level == advisor.ERROR]


def test_cuda13_on_pascal_is_an_error():
    findings = advisor.check_system(_system(P100_SMI, IVY_BRIDGE_CPUINFO, "13.0"))
    errors = [f for f in findings if f.level == advisor.ERROR]
    assert any("CUDA toolkit too new" in f.title for f in errors)


def test_driver_590_on_pascal_is_an_error():
    findings = advisor.check_system(_system(P40_SMI, MODERN_CPUINFO, "12.8"))
    assert any(f.level == advisor.ERROR and "Driver" in f.title for f in findings)


def test_missing_avx2_is_warned_about():
    findings = advisor.check_system(_system(P100_SMI, IVY_BRIDGE_CPUINFO, "12.4"))
    warns = [f for f in findings if "AVX2" in f.title]
    assert warns and "SIGILL" in warns[0].detail


def test_modern_cpu_has_no_avx2_warning():
    findings = advisor.check_system(_system(P100_SMI, MODERN_CPUINFO, "12.4"))
    assert not [f for f in findings if "AVX2" in f.title]


def test_mixed_architectures_warned():
    findings = advisor.check_system(_system(MIXED_SMI, MODERN_CPUINFO, "12.4"))
    assert any("Mixed GPU architectures" in f.title for f in findings)


# --- build recipe ----------------------------------------------------------

def test_build_recipe_targets_detected_architectures():
    recipe = "\n".join(advisor.build_recipe(_system(MIXED_SMI, MODERN_CPUINFO, "12.4")))
    assert "-DGGML_CUDA=ON" in recipe
    assert '-DCMAKE_CUDA_ARCHITECTURES="60;86"' in recipe


def test_build_recipe_forces_mmq_on_p40_only():
    p40 = "\n".join(advisor.build_recipe(_system(P40_SMI, MODERN_CPUINFO, "12.8")))
    p100 = "\n".join(advisor.build_recipe(_system(P100_SMI, MODERN_CPUINFO, "12.4")))
    assert "-DGGML_CUDA_FORCE_MMQ=ON" in p40
    assert "-DGGML_CUDA_FORCE_MMQ=ON" not in p100


def test_build_recipe_without_gpu_disables_cuda():
    recipe = "\n".join(advisor.build_recipe(_system("", MODERN_CPUINFO, None)))
    assert "-DGGML_CUDA=OFF" in recipe


# --- Apple Silicon ---------------------------------------------------------

def _mac(ram_gib: int = 8, chip: str = "Apple M1", cores: int = 8,
         perf_cores: int = 4) -> System:
    cpu = Cpu(model=chip, cores=cores, flags={"FEAT_DotProd", "FEAT_I8MM"},
              isa="arm64", perf_cores=perf_cores)
    ram = ram_gib * memory.GIB
    gpu = Gpu(index=0, name=chip, vram_mib=int(ram * 0.75) // (1024 * 1024),
              compute_cap=None, arch="Apple Silicon", unified=True,
              bandwidth_gbs=(apple_chip_for(chip).bandwidth_gbs if apple_chip_for(chip) else None),
              cores=8)
    return System(os="Darwin 23.6.0", cpu=cpu, ram_bytes=ram, gpus=[gpu],
                  driver_version=None, cuda_version=None, platform="darwin")


def test_apple_chip_matching_prefers_longest_name():
    assert apple_chip_for("Apple M1 Max").name == "M1 Max"
    assert apple_chip_for("Apple M1").name == "M1"
    assert apple_chip_for("Intel Core i9") is None


def test_arm_cpu_gets_no_avx_warnings():
    findings = advisor.check_system(_mac())
    titles = " ".join(_titles(findings))
    assert "AVX" not in titles
    assert "SIGILL" not in " ".join(f.detail for f in findings)


def test_apple_silicon_reports_unified_memory_and_thread_hint():
    findings = advisor.check_system(_mac())
    assert any("Unified memory" in f.title for f in findings)
    assert any("iogpu.wired_limit_mb" in f.fix for f in findings)
    assert any("Heterogeneous cores" in f.title for f in findings)


def test_small_mac_warns_about_tight_memory():
    assert any("Tight memory" in f.title for f in advisor.check_system(_mac(ram_gib=8)))
    assert not any("Tight memory" in f.title for f in advisor.check_system(_mac(ram_gib=64)))


def test_metal_build_recipe():
    recipe = "\n".join(advisor.build_recipe(_mac()))
    assert "-DGGML_METAL=ON" in recipe
    assert "CUDA" not in recipe


def test_feature_lookup_is_case_insensitive():
    cpu = Cpu(model="Apple M2", cores=8, flags={"FEAT_DotProd"}, isa="arm64")
    assert cpu.has("FEAT_DotProd")
    assert cpu.has("FEAT_DOTPROD")
    assert cpu.has("feat_dotprod")
    assert not cpu.has("FEAT_SME")


def test_i8mm_implies_dotprod():
    # A missing sysctl key must not produce a bogus "no dotprod" warning.
    assert implied_arm_features({"FEAT_I8MM"}) == {"FEAT_DotProd"}
    assert implied_arm_features({"FEAT_I8MM", "FEAT_DotProd"}) == set()


def test_m2_without_dotprod_key_gets_no_warning():
    mac = _mac(chip="Apple M2")
    mac.cpu.flags = {"FEAT_I8MM", "FEAT_BF16", "FEAT_FP16"}
    mac.cpu.flags |= implied_arm_features(mac.cpu.flags)
    assert not any("dotprod" in f.title for f in advisor.check_system(mac))
    assert "dotprod" in mac.cpu.simd_level


def test_parse_darwin_x86_features():
    flags = parse_darwin_x86_features("FPU VME SSE4.1 SSE4.2 AVX1.0 AVX2 FMA")
    assert flags == {"sse4_1", "sse4_2", "avx", "avx2", "fma"}


def test_parse_gpu_core_count():
    assert parse_gpu_core_count("      Total Number of Cores: 10\n") == 10
    assert parse_gpu_core_count("nothing here") is None


def test_backend_selection():
    assert _mac().backend == "metal"
    assert _system(P100_SMI, MODERN_CPUINFO, "12.4").backend == "cuda"
    assert _system("", MODERN_CPUINFO, None).backend == "cpu"


def test_metal_plan_adds_thread_flag(tiny_model):
    model = read_gguf(tiny_model)
    plan = advisor.plan_run(model, _mac(ram_gib=32), ctx=4096)
    assert plan.threads == 4
    assert "-t 4" in plan.command("m.gguf")


def test_decode_ceiling_scales_with_bandwidth():
    slow = memory.decode_ceiling_tps(4 * memory.GIB, 68.0)
    fast = memory.decode_ceiling_tps(4 * memory.GIB, 400.0)
    assert slow is not None and fast > slow
    assert memory.decode_ceiling_tps(4 * memory.GIB, None) is None


def test_metal_overhead_is_lower_than_cuda():
    assert memory.runtime_overhead(1, "metal") < memory.runtime_overhead(1, "cuda")


# --- GGUF reading ----------------------------------------------------------

def _write_gguf(path: Path, kv: dict) -> Path:
    def s(text: str) -> bytes:
        raw = text.encode()
        return struct.pack("<Q", len(raw)) + raw

    body = b""
    for key, (vtype, value) in kv.items():
        body += s(key) + struct.pack("<I", vtype)
        if vtype == 8:
            body += s(value)
        elif vtype == 4:
            body += struct.pack("<I", value)
        elif vtype == 9:  # array of uint32
            body += struct.pack("<I", 4) + struct.pack("<Q", len(value))
            body += b"".join(struct.pack("<I", v) for v in value)
        else:
            raise AssertionError(vtype)
    header = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", len(kv))
    path.write_bytes(header + body + b"\x00" * 4096)
    return path


@pytest.fixture()
def tiny_model(tmp_path: Path):
    return _write_gguf(tmp_path / "tiny-Q4_K_M.gguf", {
        "general.architecture": (8, "llama"),
        "general.name": (8, "Tiny Test"),
        "llama.block_count": (4, 32),
        "llama.embedding_length": (4, 4096),
        "llama.context_length": (4, 8192),
        "llama.attention.head_count": (4, 32),
        "llama.attention.head_count_kv": (4, 8),
        "llama.attention.key_length": (4, 128),
        "llama.vocab_size": (4, 32000),
    })


def test_read_gguf_metadata(tiny_model):
    model = read_gguf(tiny_model)
    assert model.arch == "llama"
    assert model.name == "Tiny Test"
    assert model.n_layer == 32
    assert model.head_dim_k == 128
    assert model.n_head_kv_per_layer == [8] * 32
    assert model.file_type == "Q4_K_M"
    assert model.weights_bytes == tiny_model.stat().st_size


def test_read_gguf_rejects_non_gguf(tmp_path):
    bad = tmp_path / "not.gguf"
    bad.write_bytes(b"NOPE" + b"\x00" * 32)
    with pytest.raises(GgufError):
        read_gguf(bad)


def test_hybrid_layers_with_zero_kv_heads_are_skipped(tmp_path):
    path = _write_gguf(tmp_path / "hybrid-IQ4_XS.gguf", {
        "general.architecture": (8, "hybrid"),
        "hybrid.block_count": (4, 4),
        "hybrid.embedding_length": (4, 1024),
        "hybrid.attention.head_count": (4, 8),
        "hybrid.attention.head_count_kv": (9, [4, 0, 0, 0]),
        "hybrid.attention.key_length": (4, 128),
    })
    model = read_gguf(path)
    full = memory.kv_cache_bytes(model, 1024, "f16", "f16")
    # only one of four layers holds a KV cache
    assert full == 1024 * 4 * (128 * 2 + 128 * 2)


# --- memory maths ----------------------------------------------------------

def test_kv_cache_scales_with_quantization(tiny_model):
    model = read_gguf(tiny_model)
    f16 = memory.kv_cache_bytes(model, 4096, "f16", "f16")
    q8 = memory.kv_cache_bytes(model, 4096, "q8_0", "q8_0")
    q4 = memory.kv_cache_bytes(model, 4096, "q8_0", "q4_0")
    assert f16 == 32 * 4096 * 8 * (128 * 2 + 128 * 2)
    assert q8 < f16
    assert q4 < q8


def test_max_context_grows_with_vram(tiny_model):
    model = read_gguf(tiny_model)
    small = memory.max_context(model, 2 * memory.GIB)
    large = memory.max_context(model, 4 * memory.GIB)
    assert 0 < small < large
    assert large <= model.n_ctx_train


def test_plan_run_downgrades_cache_when_vram_is_tight(tiny_model):
    model = read_gguf(tiny_model)
    tight = advisor.plan_run(model, _system(P100_SMI, IVY_BRIDGE_CPUINFO, "12.4"), ctx=8192)
    assert tight.ctx == 8192
    assert tight.n_gpu_layers == "all"
    assert "llama-server" in tight.command("/models/tiny.gguf")


def test_plan_run_offloads_partially_when_model_is_too_big(tiny_model, monkeypatch):
    model = read_gguf(tiny_model)
    monkeypatch.setattr(model, "weights_bytes", 40 * memory.GIB, raising=False)
    plan = advisor.plan_run(model, _system(P100_SMI, MODERN_CPUINFO, "12.4"), ctx=4096)
    assert plan.n_gpu_layers != "all"
    assert plan.budget.partial
    # The budget must describe the plan being recommended: only the offloaded
    # layers occupy VRAM, and the remainder is reported separately.
    assert plan.budget.weights < 40 * memory.GIB
    assert plan.budget.weights_cpu > 0
    assert plan.budget.weights + plan.budget.weights_cpu == 40 * memory.GIB
    assert plan.budget.fits
    assert any("Partial offload" in note for note in plan.notes)


def test_partial_plan_suggests_a_context_that_fits_fully(tiny_model, monkeypatch):
    model = read_gguf(tiny_model)
    monkeypatch.setattr(model, "weights_bytes", 14 * memory.GIB, raising=False)
    plan = advisor.plan_run(model, _system(P100_SMI, MODERN_CPUINFO, "12.4"), ctx=200000)
    assert plan.budget.partial
    assert any("would fit entirely on the GPU" in note for note in plan.notes)


def test_split_weights_accounts_for_the_output_layer(tiny_model):
    model = read_gguf(tiny_model)
    full, none_ = memory.split_weights(model, None)
    assert (full, none_) == (model.weights_bytes, 0)
    assert memory.split_weights(model, 999) == (model.weights_bytes, 0)
    on_gpu, on_cpu = memory.split_weights(model, 16)  # half of 32 layers
    assert on_gpu + on_cpu == model.weights_bytes
    assert on_gpu == int(model.weights_bytes * 16 / 33)
    assert memory.split_weights(model, 0) == (0, model.weights_bytes)
