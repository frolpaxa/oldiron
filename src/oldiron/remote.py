"""Read GGUF metadata over HTTP, without downloading the weights.

A GGUF file starts with its metadata, so a couple of range requests are enough
to answer "will this fit?" for a 15 GB model. Standard library only.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from . import __version__
from .gguf_meta import SHARD_RE, GgufError, GgufModel, parse_header

HF_HOST = "huggingface.co"
DEFAULT_QUANT = "Q4_K_M"  # what llama.cpp's -hf falls back to
USER_AGENT = f"oldiron/{__version__} (+https://github.com/frolpaxa/oldiron)"

FIRST_CHUNK = 1024 * 1024
MAX_CHUNK = 8 * 1024 * 1024
MAX_HEADER_BYTES = 64 * 1024 * 1024  # a tokenizer block is big, but not this big

_BLOB_URL_RE = re.compile(
    r"^https?://(?:www\.)?huggingface\.co/(?P<repo>[^/]+/[^/]+)/"
    r"(?:blob|resolve)/(?P<revision>[^/]+)/(?P<path>.+?)(?:\?.*)?$"
)


class RemoteError(Exception):
    pass


@dataclass
class HfRef:
    repo: str
    quant: str | None = None
    revision: str = "main"
    filename: str | None = None  # set when the spec named an exact file

    def __str__(self) -> str:
        return f"{self.repo}:{self.quant}" if self.quant else self.repo


def is_remote_spec(spec: str) -> bool:
    return spec.startswith("hf:") or bool(_BLOB_URL_RE.match(spec))


def parse_spec(spec: str) -> HfRef:
    """Accept `hf:user/repo[:QUANT]` or a huggingface.co blob/resolve URL."""
    match = _BLOB_URL_RE.match(spec)
    if match:
        return HfRef(
            repo=match.group("repo"),
            revision=urllib.parse.unquote(match.group("revision")),
            filename=urllib.parse.unquote(match.group("path")),
        )

    body = spec[3:] if spec.startswith("hf:") else spec
    body = body.strip("/")
    revision = "main"
    if "@" in body:
        body, revision = body.rsplit("@", 1)
    quant = None
    parts = body.split(":")
    if len(parts) == 2:
        body, quant = parts
    elif len(parts) > 2:
        raise RemoteError(f"cannot parse model spec: {spec!r}")
    if body.count("/") != 1 or not all(body.split("/")):
        raise RemoteError(
            f"expected hf:<user>/<repo>[:QUANT], got {spec!r}"
        )
    return HfRef(repo=body, quant=quant or None, revision=revision)


def _request(url: str, token: str | None, headers: dict | None = None):
    head = {"User-Agent": USER_AGENT}
    if headers:
        head.update(headers)
    if token:
        head["Authorization"] = f"Bearer {token}"
    return urllib.request.Request(url, headers=head)


def _open(url: str, token: str | None, headers: dict | None = None, timeout: int = 30):
    try:
        return urllib.request.urlopen(_request(url, token, headers), timeout=timeout)
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise RemoteError(
                f"{url} returned {exc.code}. The repository may be gated or private; "
                "set HF_TOKEN or pass --hf-token."
            ) from exc
        if exc.code == 404:
            raise RemoteError(f"not found: {url}") from exc
        raise RemoteError(f"HTTP {exc.code} for {url}") from exc
    except urllib.error.URLError as exc:
        raise RemoteError(f"cannot reach {url}: {exc.reason}") from exc


def token_from_env() -> str | None:
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(name)
        if value:
            return value.strip()
    path = Path.home() / ".cache" / "huggingface" / "token"
    try:
        return path.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


# --- repository listing ----------------------------------------------------

@dataclass
class RemoteFile:
    path: str
    size: int


def parse_tree(payload: list) -> list[RemoteFile]:
    """Pull .gguf files out of the Hub tree listing.

    LFS entries carry the real size under `lfs.size`; the plain `size` field can
    be the pointer size instead, which would make every model look tiny.
    """
    files: list[RemoteFile] = []
    for entry in payload:
        if entry.get("type") != "file":
            continue
        path = entry.get("path") or ""
        if not path.lower().endswith(".gguf"):
            continue
        lfs = entry.get("lfs") or {}
        size = lfs.get("size") or entry.get("size") or 0
        files.append(RemoteFile(path=path, size=int(size)))
    return sorted(files, key=lambda f: f.path)


def list_gguf_files(ref: HfRef, token: str | None) -> list[RemoteFile]:
    url = (f"https://{HF_HOST}/api/models/{ref.repo}/tree/"
           f"{urllib.parse.quote(ref.revision)}?recursive=true")
    with _open(url, token) as response:
        payload = json.load(response)
    if not isinstance(payload, list):
        raise RemoteError(f"unexpected listing for {ref.repo}")
    files = parse_tree(payload)
    if not files:
        raise RemoteError(f"no .gguf files in {ref.repo} at {ref.revision}")
    return files


def shard_group(files: list[RemoteFile], chosen: str) -> list[RemoteFile]:
    """Every part of a multi-file model, so the weight total is the real one."""
    match = SHARD_RE.match(Path(chosen).name)
    if not match:
        return [f for f in files if f.path == chosen]
    prefix = f"{Path(chosen).parent}/" if Path(chosen).parent != Path(".") else ""
    stem, total = match.group("stem"), match.group("total")
    wanted = {f"{prefix}{stem}-{i:05d}-of-{total}.gguf"
              for i in range(1, int(total) + 1)}
    return [f for f in files if f.path in wanted]


def pick_file(files: list[RemoteFile], quant: str | None) -> str:
    """Choose which file in the repo to inspect."""
    candidates = files
    if quant:
        wanted = quant.lower()
        candidates = [f for f in files if wanted in Path(f.path).name.lower()]
        if not candidates:
            candidates = [f for f in files if wanted in f.path.lower()]
        if not candidates:
            available = sorted({_quant_of(f.path) for f in files} - {""})
            raise RemoteError(
                f"no file matching {quant!r}. Available: {', '.join(available) or 'unknown'}"
            )
    else:
        default = [f for f in files if DEFAULT_QUANT.lower() in Path(f.path).name.lower()]
        candidates = default or files

    # For a sharded model, the first shard holds the metadata.
    shards = [f for f in candidates if SHARD_RE.match(Path(f.path).name)]
    if shards and len(shards) == len(candidates):
        first = [f for f in shards if "-00001-of-" in Path(f.path).name]
        return (first or shards)[0].path
    plain = [f for f in candidates if not SHARD_RE.match(Path(f.path).name)]
    return (plain or candidates)[0].path


def _quant_of(path: str) -> str:
    match = re.search(r"(IQ\d[A-Z_]*|Q\d[_A-Z0-9]*|BF16|F16|F32)", Path(path).name, re.I)
    return match.group(1).upper() if match else ""


# --- range-request stream --------------------------------------------------

class RangeStream:
    """A read-only, forward-only stream backed by HTTP range requests."""

    def __init__(self, url: str, token: str | None = None, opener=None):
        self.url = url
        self.token = token
        self._opener = opener or (lambda u, t, h: _open(u, t, h))
        self._buf = b""
        self._buf_pos = 0
        self._next_offset = 0
        self._chunk = FIRST_CHUNK
        self._fetched = 0
        self.requests = 0

    def _fetch(self, need: int) -> None:
        size = max(need, self._chunk)
        if self._fetched + size > MAX_HEADER_BYTES:
            raise GgufError("metadata block larger than expected; giving up")
        start = self._next_offset
        end = start + size - 1
        response = self._opener(self.url, self.token, {"Range": f"bytes={start}-{end}"})
        self.requests += 1
        with response:
            status = getattr(response, "status", None) or response.getcode()
            data = response.read()
            if status == 200 and start:
                # Server ignored the Range header and sent the whole file.
                if not response.headers.get("Content-Range"):
                    data = data[start:]
        if not data:
            raise GgufError("server returned no data for the requested range")
        self._buf = self._buf[self._buf_pos:] + data
        self._buf_pos = 0
        self._next_offset = start + len(data)
        self._fetched += len(data)
        self._chunk = min(self._chunk * 2, MAX_CHUNK)

    def read(self, n: int) -> bytes:
        while len(self._buf) - self._buf_pos < n:
            self._fetch(n - (len(self._buf) - self._buf_pos))
        chunk = self._buf[self._buf_pos:self._buf_pos + n]
        self._buf_pos += n
        return chunk


def resolve_url(ref: HfRef, filename: str) -> str:
    return (f"https://{HF_HOST}/{ref.repo}/resolve/"
            f"{urllib.parse.quote(ref.revision)}/{urllib.parse.quote(filename)}")


def read_gguf_remote(spec: str, token: str | None = None) -> GgufModel:
    """Build a GgufModel from a Hugging Face repo without downloading weights."""
    ref = parse_spec(spec)
    token = token or token_from_env()
    files = list_gguf_files(ref, token)

    chosen = ref.filename or pick_file(files, ref.quant)
    if ref.filename and not any(f.path == chosen for f in files):
        raise RemoteError(f"{chosen} is not in {ref.repo} at {ref.revision}")

    group = shard_group(files, chosen) or [RemoteFile(chosen, 0)]
    weights = sum(f.size for f in group)

    stream = RangeStream(resolve_url(ref, chosen), token)
    version, tensor_count, kv = parse_header(stream, Path(chosen).name)

    quant = ref.quant or _quant_of(chosen) or None
    return GgufModel(
        path=Path(chosen),
        version=version,
        tensor_count=tensor_count,
        kv=kv,
        weights_bytes=weights,
        shard_paths=[Path(f.path) for f in group],
        remote_ref=f"{ref.repo}:{quant}" if quant else ref.repo,
    )
