"""Memory arithmetic: what actually has to fit in VRAM."""

from __future__ import annotations

from dataclasses import dataclass

from .gguf_meta import GgufModel

GIB = 1024 ** 3
MIB = 1024 ** 2

# Bytes per element for llama.cpp KV cache types (block size 32 for the
# quantized ones, so q8_0 is 34 bytes per 32 values).
CACHE_TYPES: dict[str, float] = {
    "f32": 4.0,
    "f16": 2.0,
    "bf16": 2.0,
    "q8_0": 34 / 32,
    "q5_1": 24 / 32,
    "q5_0": 22 / 32,
    "q4_1": 20 / 32,
    "q4_0": 18 / 32,
    "iq4_nl": 18 / 32,
}

QUANTIZED_CACHE = {"q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "iq4_nl"}


@dataclass
class Budget:
    weights: int  # the part that lives in GPU memory
    kv_cache: int
    compute: int
    overhead: int
    available: int
    weights_cpu: int = 0  # layers left on the host, when offload is partial

    @property
    def total(self) -> int:
        """What has to fit on the device. Host-side weights are excluded."""
        return self.weights + self.kv_cache + self.compute + self.overhead

    @property
    def partial(self) -> bool:
        return self.weights_cpu > 0

    @property
    def headroom(self) -> int:
        return self.available - self.total

    @property
    def fits(self) -> bool:
        return self.headroom >= 0


def kv_cache_bytes(model: GgufModel, ctx: int, ctk: str = "f16", ctv: str = "f16") -> int:
    """KV cache size for `ctx` tokens.

    Layers whose head_count_kv is 0 (linear-attention layers in hybrid models)
    contribute nothing, which is why the per-layer list is used rather than a
    single number times the layer count.
    """
    bk, bv = CACHE_TYPES[ctk], CACHE_TYPES[ctv]
    k_dim, v_dim = model.head_dim_k, model.head_dim_v
    total = 0.0
    for heads in model.n_head_kv_per_layer:
        if heads <= 0:
            continue
        total += ctx * heads * (k_dim * bk + v_dim * bv)
    return int(total)


def compute_buffer_bytes(model: GgufModel, ubatch: int = 512) -> int:
    """Rough size of the graph/compute buffers. Estimate, not a promise."""
    n_embd = model.n_embd or 4096
    vocab = model.n_vocab or 32000
    per_token = n_embd * 12 * 4  # activations kept live in the graph
    logits = vocab * 4 * min(ubatch, 8)
    return int(ubatch * per_token + logits + 256 * MIB)


def runtime_overhead(gpu_count: int = 1, backend: str = "cuda") -> int:
    """Driver context plus a safety margin.

    A CUDA context costs a few hundred MiB per device. Metal has no comparable
    per-device context, and the OS headroom is already excluded by the 75% cap.
    """
    if backend == "metal":
        return 384 * MIB
    if backend == "cpu":
        return 256 * MIB
    return 400 * MIB * max(gpu_count, 1) + 512 * MIB


def split_weights(model: GgufModel, gpu_layers: int | None) -> tuple[int, int]:
    """Weight bytes on the device and on the host for a given -ngl.

    `n_layer + 1` because the output/embedding tensors are offloaded as one more
    unit on top of the transformer blocks.
    """
    total = model.weights_bytes
    n_layer = model.n_layer or 1
    if gpu_layers is None or gpu_layers >= n_layer:
        return total, 0
    if gpu_layers <= 0:
        return 0, total
    on_gpu = int(total * gpu_layers / (n_layer + 1))
    return on_gpu, total - on_gpu


def plan(
    model: GgufModel,
    vram_bytes: int,
    ctx: int,
    ctk: str = "f16",
    ctv: str = "f16",
    ubatch: int = 512,
    gpu_count: int = 1,
    backend: str = "cuda",
    gpu_layers: int | None = None,
) -> Budget:
    """Device memory required. With `gpu_layers`, only the offloaded part counts."""
    on_gpu, on_cpu = split_weights(model, gpu_layers)
    return Budget(
        weights=on_gpu,
        weights_cpu=on_cpu,
        kv_cache=kv_cache_bytes(model, ctx, ctk, ctv),
        compute=compute_buffer_bytes(model, ubatch),
        overhead=runtime_overhead(gpu_count, backend),
        available=vram_bytes,
    )


def max_context(
    model: GgufModel,
    vram_bytes: int,
    ctk: str = "f16",
    ctv: str = "f16",
    ubatch: int = 512,
    gpu_count: int = 1,
    step: int = 1024,
    backend: str = "cuda",
) -> int:
    """Largest context (rounded down to `step`) that still fits fully offloaded."""
    per_token = kv_cache_bytes(model, 1, ctk, ctv)
    if per_token <= 0:
        return 0
    fixed = plan(model, vram_bytes, 0, ctk, ctv, ubatch, gpu_count, backend)
    free = vram_bytes - fixed.total
    if free <= 0:
        return 0
    ctx = int(free // per_token)
    ctx -= ctx % step
    if model.n_ctx_train:
        ctx = min(ctx, model.n_ctx_train)
    return max(ctx, 0)


def layers_that_fit(model: GgufModel, vram_bytes: int, ctx: int, ctk: str, ctv: str,
                    ubatch: int = 512, gpu_count: int = 1, backend: str = "cuda") -> int:
    """How many layers to offload when the whole model does not fit."""
    n_layer = model.n_layer or 1
    per_layer = model.weights_bytes / (n_layer + 1)
    fixed = compute_buffer_bytes(model, ubatch) + runtime_overhead(gpu_count, backend)
    kv = kv_cache_bytes(model, ctx, ctk, ctv)
    free = vram_bytes - fixed - kv
    if free <= 0:
        return 0
    return max(0, min(n_layer, int(free // per_layer)))


def decode_ceiling_tps(weights_bytes: int, bandwidth_gbs: float | None) -> float | None:
    """Upper bound on tokens/s: every weight is read once per generated token.

    Real throughput lands at roughly 60-80% of this, which is still the number
    that decides whether a model is usable or merely loadable.
    """
    if not bandwidth_gbs or weights_bytes <= 0:
        return None
    return bandwidth_gbs * 1e9 / weights_bytes


def human(num_bytes: float) -> str:
    if num_bytes >= GIB:
        return f"{num_bytes / GIB:.2f} GiB"
    return f"{num_bytes / MIB:.0f} MiB"
