"""A minimal, dependency-free GGUF metadata reader.

Only the header is read: the key/value block at the start of the file. Tensor
data is never touched, so this is fast even for a 30 GB shard on a slow disk.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from pathlib import Path

MAGIC = b"GGUF"

(
    T_UINT8, T_INT8, T_UINT16, T_INT16, T_UINT32, T_INT32,
    T_FLOAT32, T_BOOL, T_STRING, T_ARRAY, T_UINT64, T_INT64, T_FLOAT64,
) = range(13)

_SCALARS = {
    T_UINT8: ("<B", 1), T_INT8: ("<b", 1),
    T_UINT16: ("<H", 2), T_INT16: ("<h", 2),
    T_UINT32: ("<I", 4), T_INT32: ("<i", 4),
    T_FLOAT32: ("<f", 4), T_BOOL: ("<?", 1),
    T_UINT64: ("<Q", 8), T_INT64: ("<q", 8), T_FLOAT64: ("<d", 8),
}

SHARD_RE = re.compile(r"^(?P<stem>.+)-(?P<idx>\d{5})-of-(?P<total>\d{5})\.gguf$", re.I)


class GgufError(Exception):
    pass


class _Reader:
    def __init__(self, fh):
        self.fh = fh

    def raw(self, n: int) -> bytes:
        data = self.fh.read(n)
        if len(data) != n:
            raise GgufError("unexpected end of file while reading metadata")
        return data

    def scalar(self, vtype: int):
        fmt, size = _SCALARS[vtype]
        return struct.unpack(fmt, self.raw(size))[0]

    def string(self) -> str:
        length = struct.unpack("<Q", self.raw(8))[0]
        if length > 64 * 1024 * 1024:
            raise GgufError("implausible string length in metadata")
        return self.raw(length).decode("utf-8", errors="replace")

    def value(self, vtype: int, keep_array: bool = True):
        if vtype in _SCALARS:
            return self.scalar(vtype)
        if vtype == T_STRING:
            return self.string()
        if vtype == T_ARRAY:
            elem_type = struct.unpack("<I", self.raw(4))[0]
            count = struct.unpack("<Q", self.raw(8))[0]
            # Token lists have hundreds of thousands of strings; we only ever
            # need their length, so large arrays are counted, not collected.
            keep = keep_array and count <= 4096
            items = []
            for _ in range(count):
                item = self.value(elem_type, keep_array=False)
                if keep:
                    items.append(item)
            return items if keep else _ArrayLength(count, elem_type)
        raise GgufError(f"unknown GGUF value type {vtype}")


@dataclass(frozen=True)
class _ArrayLength:
    count: int
    elem_type: int

    def __len__(self) -> int:
        return self.count


@dataclass
class GgufModel:
    path: Path
    version: int
    tensor_count: int
    kv: dict
    weights_bytes: int
    shard_paths: list[Path]
    remote_ref: str | None = None  # "<user>/<repo>:QUANT" when read over HTTP

    @property
    def launch_arg(self) -> str:
        """How llama-server should be pointed at this model."""
        if self.remote_ref:
            return f"-hf {self.remote_ref}"
        return f"-m {self.path}"

    # --- convenience accessors -------------------------------------------------
    @property
    def arch(self) -> str:
        return str(self.kv.get("general.architecture", "unknown"))

    @property
    def name(self) -> str:
        return str(self.kv.get("general.name", self.path.stem))

    def a(self, suffix: str, default=None):
        """Read an architecture-scoped key, e.g. a('block_count')."""
        return self.kv.get(f"{self.arch}.{suffix}", default)

    @property
    def n_layer(self) -> int:
        return int(self.a("block_count", 0) or 0)

    @property
    def n_embd(self) -> int:
        return int(self.a("embedding_length", 0) or 0)

    @property
    def n_head(self) -> int:
        value = self.a("attention.head_count", 0)
        if isinstance(value, list) and value:
            return int(max(value))
        return int(value or 0)

    @property
    def n_head_kv_per_layer(self) -> list[int]:
        value = self.a("attention.head_count_kv")
        if isinstance(value, list) and value:
            return [int(v) for v in value]
        if value is None:
            value = self.n_head
        return [int(value)] * max(self.n_layer, 0)

    @property
    def head_dim_k(self) -> int:
        value = self.a("attention.key_length")
        if value:
            return int(value)
        return self.n_embd // self.n_head if self.n_head else 0

    @property
    def head_dim_v(self) -> int:
        value = self.a("attention.value_length")
        if value:
            return int(value)
        return self.head_dim_k

    @property
    def n_ctx_train(self) -> int:
        return int(self.a("context_length", 0) or 0)

    @property
    def sliding_window(self) -> int:
        return int(self.a("attention.sliding_window", 0) or 0)

    @property
    def n_vocab(self) -> int:
        tokens = self.kv.get("tokenizer.ggml.tokens")
        if tokens is not None:
            return len(tokens)
        return int(self.a("vocab_size", 0) or 0)

    @property
    def file_type(self) -> str:
        match = re.search(r"(IQ\d[A-Z_]*|Q\d[_A-Z0-9]*|BF16|F16|F32)", self.path.name, re.I)
        return match.group(1).upper() if match else "unknown"

    @property
    def is_moe(self) -> bool:
        return bool(self.a("expert_count", 0))


def _sibling_shards(path: Path) -> list[Path]:
    match = SHARD_RE.match(path.name)
    if not match:
        return [path]
    stem, total = match.group("stem"), int(match.group("total"))
    shards = []
    for i in range(1, total + 1):
        candidate = path.with_name(f"{stem}-{i:05d}-of-{total:05d}.gguf")
        if candidate.exists():
            shards.append(candidate)
    return shards or [path]


def parse_header(stream, label: str = "file") -> tuple[int, int, dict]:
    """Read magic, version, counts and the key/value block from any byte stream.

    The stream only has to provide `read(n)`, which is what lets the same parser
    work on a local file and on HTTP range requests.
    """
    reader = _Reader(stream)
    if reader.raw(4) != MAGIC:
        raise GgufError(f"{label} is not a GGUF file (bad magic)")
    version = struct.unpack("<I", reader.raw(4))[0]
    if version < 2 or version > 4:
        raise GgufError(f"unsupported GGUF version {version}")
    tensor_count = struct.unpack("<Q", reader.raw(8))[0]
    kv_count = struct.unpack("<Q", reader.raw(8))[0]
    if kv_count > 1_000_000:
        raise GgufError("implausible metadata key count")
    kv: dict = {}
    for _ in range(kv_count):
        key = reader.string()
        vtype = struct.unpack("<I", reader.raw(4))[0]
        kv[key] = reader.value(vtype)
    return version, tensor_count, kv


def read_gguf(path: str | Path) -> GgufModel:
    path = Path(path)
    with path.open("rb") as fh:
        version, tensor_count, kv = parse_header(fh, path.name)

    shards = _sibling_shards(path)
    return GgufModel(
        path=path,
        version=version,
        tensor_count=tensor_count,
        kv=kv,
        weights_bytes=sum(p.stat().st_size for p in shards),
        shard_paths=shards,
    )
