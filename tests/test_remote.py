"""Tests for reading GGUF metadata over HTTP."""

from __future__ import annotations

import json
import os
import struct
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from oldiron import remote
from oldiron.gguf_meta import GgufError, read_gguf
from oldiron.remote import (
    HfRef,
    RangeStream,
    RemoteError,
    RemoteFile,
    is_remote_spec,
    parse_spec,
    parse_tree,
    pick_file,
    resolve_url,
    shard_group,
)


# --- spec parsing ----------------------------------------------------------

def test_is_remote_spec():
    assert is_remote_spec("hf:unsloth/Qwen3-4B-GGUF")
    assert is_remote_spec("https://huggingface.co/unsloth/Qwen3-4B-GGUF/blob/main/a.gguf")
    assert not is_remote_spec("/models/a.gguf")
    assert not is_remote_spec("./a.gguf")


def test_parse_spec_variants():
    assert parse_spec("hf:unsloth/Qwen3-4B-GGUF") == HfRef("unsloth/Qwen3-4B-GGUF")
    assert parse_spec("hf:unsloth/Qwen3-4B-GGUF:Q4_K_M").quant == "Q4_K_M"
    assert parse_spec("hf:unsloth/Qwen3-4B-GGUF:Q4_K_M@refs/pr/1").revision == "refs/pr/1"


def test_parse_spec_url_keeps_exact_filename():
    ref = parse_spec(
        "https://huggingface.co/unsloth/Qwen3-4B-GGUF/blob/main/Qwen3-4B-Q4_K_M.gguf"
    )
    assert ref.repo == "unsloth/Qwen3-4B-GGUF"
    assert ref.filename == "Qwen3-4B-Q4_K_M.gguf"
    assert ref.revision == "main"


def test_parse_spec_rejects_nonsense():
    with pytest.raises(RemoteError):
        parse_spec("hf:justaname")
    with pytest.raises(RemoteError):
        parse_spec("hf:a/b:c:d")


def test_resolve_url_quotes_path():
    url = resolve_url(HfRef("user/repo"), "sub dir/model.gguf")
    assert url == "https://huggingface.co/user/repo/resolve/main/sub%20dir/model.gguf"


# --- listing ---------------------------------------------------------------

TREE = [
    {"type": "directory", "path": "Q8_0"},
    {"type": "file", "path": "README.md", "size": 1024},
    {"type": "file", "path": "Qwen3-4B-Q4_K_M.gguf", "size": 135,
     "lfs": {"size": 2_600_000_000}},
    {"type": "file", "path": "Qwen3-4B-Q8_0.gguf", "size": 4_300_000_000},
    {"type": "file", "path": "Q8_0/big-00001-of-00002.gguf", "size": 1_000},
    {"type": "file", "path": "Q8_0/big-00002-of-00002.gguf", "size": 2_000},
]


def test_parse_tree_prefers_lfs_size():
    files = parse_tree(TREE)
    paths = {f.path: f.size for f in files}
    assert "README.md" not in paths
    # the pointer size (135) must not be mistaken for the real size
    assert paths["Qwen3-4B-Q4_K_M.gguf"] == 2_600_000_000
    assert paths["Qwen3-4B-Q8_0.gguf"] == 4_300_000_000


def test_pick_file_by_quant_and_default():
    files = parse_tree(TREE)
    assert pick_file(files, "q4_k_m") == "Qwen3-4B-Q4_K_M.gguf"
    assert pick_file(files, None) == "Qwen3-4B-Q4_K_M.gguf"  # DEFAULT_QUANT
    with pytest.raises(RemoteError) as excinfo:
        pick_file(files, "IQ2_XXS")
    assert "Available" in str(excinfo.value)


def test_pick_file_returns_first_shard():
    files = parse_tree(TREE)
    assert pick_file(files, "Q8_0/big") == "Q8_0/big-00001-of-00002.gguf"


def test_shard_group_sums_all_parts():
    files = parse_tree(TREE)
    group = shard_group(files, "Q8_0/big-00001-of-00002.gguf")
    assert sum(f.size for f in group) == 3_000
    single = shard_group(files, "Qwen3-4B-Q8_0.gguf")
    assert [f.path for f in single] == ["Qwen3-4B-Q8_0.gguf"]


# --- range stream against a real server ------------------------------------

@pytest.fixture(scope="module")
def http_server(tmp_path_factory):
    """A real HTTP server; SimpleHTTPRequestHandler ignores Range headers."""
    root = tmp_path_factory.mktemp("srv")
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=str(root))
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", root
    server.shutdown()
    server.server_close()


class _RangeHandler(SimpleHTTPRequestHandler):
    """Serves byte ranges properly, the way the Hub CDN does."""

    def do_GET(self):  # noqa: N802
        path = Path(self.directory) / self.path.lstrip("/")
        if not path.is_file():
            self.send_error(404)
            return
        data = path.read_bytes()
        header = self.headers.get("Range")
        if not header:
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        start, _, end = header.replace("bytes=", "").partition("-")
        start = int(start)
        stop = min(int(end) + 1, len(data)) if end else len(data)
        chunk = data[start:stop]
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {start}-{stop - 1}/{len(data)}")
        self.send_header("Content-Length", str(len(chunk)))
        self.end_headers()
        self.wfile.write(chunk)

    def log_message(self, *args):
        pass


@pytest.fixture(scope="module")
def range_server(tmp_path_factory):
    root = tmp_path_factory.mktemp("range")
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(_RangeHandler, directory=str(root))
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", root
    server.shutdown()
    server.server_close()


def _make_gguf(path: Path, vocab_size: int = 0, tail_bytes: int = 4096) -> Path:
    """Write a valid GGUF header, optionally with a big token array."""
    def s(text: str) -> bytes:
        raw = text.encode()
        return struct.pack("<Q", len(raw)) + raw

    kv = [
        ("general.architecture", 8, "llama"),
        ("general.name", 8, "Remote Test"),
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
    if vocab_size:
        # array of strings, the part that makes real headers megabytes long
        body += s("tokenizer.ggml.tokens") + struct.pack("<I", 9)
        body += struct.pack("<I", 8) + struct.pack("<Q", vocab_size)
        for i in range(vocab_size):
            body += s(f"token_{i:06d}")
    count = len(kv) + (1 if vocab_size else 0)
    head = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", count)
    path.write_bytes(head + body + b"\x00" * tail_bytes)
    return path


def test_range_stream_reads_incrementally(range_server, monkeypatch):
    base, root = range_server
    _make_gguf(root / "small.gguf")
    monkeypatch.setattr(remote, "FIRST_CHUNK", 64)
    monkeypatch.setattr(remote, "MAX_CHUNK", 256)
    stream = RangeStream(f"{base}/small.gguf")
    assert stream.read(4) == b"GGUF"
    assert struct.unpack("<I", stream.read(4))[0] == 3
    assert stream.requests >= 1


def test_remote_header_matches_local_parse(range_server, monkeypatch):
    base, root = range_server
    # 8 MiB of "weights" after the header: the whole point is not to fetch them.
    path = _make_gguf(root / "big.gguf", vocab_size=20000, tail_bytes=8 * 1024 * 1024)
    monkeypatch.setattr(remote, "FIRST_CHUNK", 8192)  # force several round trips

    local = read_gguf(path)
    stream = RangeStream(f"{base}/big.gguf")
    from oldiron.gguf_meta import parse_header
    version, tensors, kv = parse_header(stream, "big.gguf")

    assert version == local.version
    assert tensors == local.tensor_count
    assert kv["general.architecture"] == "llama"
    assert len(kv["tokenizer.ggml.tokens"]) == 20000
    assert stream.requests > 1  # proves the progressive fetch path is exercised
    # only the header was transferred, not the 8 MiB body
    assert stream._fetched < path.stat().st_size / 4


def test_stream_survives_server_ignoring_range(http_server, monkeypatch):
    base, root = http_server
    _make_gguf(root / "plain.gguf", vocab_size=500)
    monkeypatch.setattr(remote, "FIRST_CHUNK", 1024)
    stream = RangeStream(f"{base}/plain.gguf")
    from oldiron.gguf_meta import parse_header
    _, _, kv = parse_header(stream, "plain.gguf")
    assert kv["general.name"] == "Remote Test"


def test_stream_refuses_runaway_metadata(range_server, monkeypatch):
    base, root = range_server
    _make_gguf(root / "capped.gguf", vocab_size=100)
    monkeypatch.setattr(remote, "MAX_HEADER_BYTES", 16)
    stream = RangeStream(f"{base}/capped.gguf")
    with pytest.raises(GgufError):
        stream.read(64)


def test_read_gguf_remote_end_to_end(range_server, monkeypatch):
    """Full path with the Hub API and download URLs pointed at the local server."""
    base, root = range_server
    _make_gguf(root / "Model-Q4_K_M.gguf", vocab_size=2000)
    size = (root / "Model-Q4_K_M.gguf").stat().st_size

    tree = [
        {"type": "file", "path": "Model-Q4_K_M.gguf", "size": 135, "lfs": {"size": size}},
        {"type": "file", "path": "Model-Q8_0.gguf", "size": size * 2},
    ]
    monkeypatch.setattr(remote, "list_gguf_files", lambda ref, token: parse_tree(tree))
    monkeypatch.setattr(remote, "resolve_url",
                        lambda ref, filename: f"{base}/{filename}")
    monkeypatch.setattr(remote, "token_from_env", lambda: None)

    model = remote.read_gguf_remote("hf:acme/Model-GGUF:Q4_K_M")
    assert model.arch == "llama"
    assert model.n_layer == 32
    assert model.weights_bytes == size
    assert model.remote_ref == "acme/Model-GGUF:Q4_K_M"
    assert model.launch_arg == "-hf acme/Model-GGUF:Q4_K_M"


def test_token_from_env(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "  secret  ")
    assert remote.token_from_env() == "secret"
    monkeypatch.delenv("HF_TOKEN")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "other")
    assert remote.token_from_env() == "other"
