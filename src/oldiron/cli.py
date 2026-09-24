"""oldiron command line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__, bench
from .advisor import ERROR, OK, WARN, build_recipe, check_system, plan_run
from .gguf_meta import GgufError, read_gguf
from .hardware import detect
from .memory import GIB, max_context
from .remote import RemoteError, is_remote_spec, read_gguf_remote
from .render import get_renderer

CACHE_COMBOS = (("f16", "f16"), ("q8_0", "q8_0"), ("q8_0", "q4_0"), ("q4_0", "q4_0"))
FOOTER = ("Numbers are estimates (+-15%). llama-server's own --fit will do the final "
          "adjustment at load time.")


def _renderer(args):
    return get_renderer(plain=getattr(args, "plain", False))


def _load_model(spec: str, ui, token: str | None = None):
    if is_remote_spec(spec):
        try:
            return read_gguf_remote(spec, token)
        except (RemoteError, GgufError) as exc:
            ui.error(str(exc))
            return None
    try:
        return read_gguf(spec)
    except (GgufError, OSError) as exc:
        ui.error(str(exc))
        return None


def cmd_doctor(args) -> int:
    system = detect()
    findings = check_system(system)
    recipe = build_recipe(system)

    if args.json:
        print(json.dumps({
            "system": system.to_dict(),
            "findings": [f.__dict__ for f in findings],
            "build": recipe,
        }, indent=2))
        return 0

    ui = _renderer(args)
    ui.system(system)
    ui.findings(findings)
    ui.build_recipe(recipe)
    return 1 if any(f.level == ERROR for f in findings) else 0


def cmd_fit(args) -> int:
    ui = _renderer(args)
    model = _load_model(args.model, ui, getattr(args, "hf_token", None))
    if model is None:
        return 2
    system = detect()
    plan = plan_run(model, system, ctx=args.ctx, ubatch=args.ubatch)
    budget = plan.budget

    if args.json:
        print(json.dumps({
            "model": {
                "name": model.name, "arch": model.arch, "quant": model.file_type,
                "layers": model.n_layer, "ctx_train": model.n_ctx_train,
                "weights_bytes": model.weights_bytes,
                "remote_ref": model.remote_ref,
            },
            "plan": {
                "ctx": plan.ctx, "cache_type_k": plan.ctk, "cache_type_v": plan.ctv,
                "n_gpu_layers": plan.n_gpu_layers, "fits": budget.fits,
                "partial_offload": budget.partial,
                "threads": plan.threads, "decode_tps_ceiling": plan.decode_tps,
                "backend": system.backend,
                "weights": budget.weights, "weights_cpu": budget.weights_cpu,
                "kv_cache": budget.kv_cache,
                "compute": budget.compute, "overhead": budget.overhead,
                "available": budget.available, "headroom": budget.headroom,
            },
            "command": plan.command(model.launch_arg),
            "notes": plan.notes,
        }, indent=2))
        return 0 if budget.fits else 1

    ui.model(model)
    ui.system(system)
    ui.plan(plan, system)
    if not args.no_table:
        rows = [
            (f"{ctk}/{ctv}",
             max_context(model, system.total_vram_bytes, ctk, ctv, args.ubatch,
                         max(len(system.gpus), 1), backend=system.backend))
            for ctk, ctv in CACHE_COMBOS
        ]
        ui.context_table(rows)
    ui.notes(plan.notes)
    ui.command(plan.command(model.launch_arg))
    ui.footer(FOOTER)
    return 0 if budget.fits else 1


def cmd_bench(args) -> int:
    ui = _renderer(args)
    model = _load_model(args.model, ui)
    if model is None:
        return 2
    system = detect()
    plan = plan_run(model, system, ctx=args.ctx, ubatch=args.ubatch)

    try:
        binary = bench.find_binary(args.llama_bench)
        result = bench.measure(model, system, plan, binary,
                               prompt_tokens=args.prompt_tokens,
                               gen_tokens=args.gen_tokens,
                               repetitions=args.repetitions)
    except bench.BenchError as exc:
        ui.error(str(exc))
        return 2

    record = bench.submission(model, system, plan, result)
    saved = None
    if not args.no_save:
        target = Path(args.output) if args.output else (
            Path("results") / bench.suggested_filename(model, system))
        saved = bench.write_submission(record, target)

    if args.json:
        print(json.dumps(record, indent=2))
        return 0

    ui.model(model)
    ui.system(system)
    ui.bench(result, saved)
    return 0


def cmd_info(args) -> int:
    ui = _renderer(args)
    model = _load_model(args.model, ui, getattr(args, "hf_token", None))
    if model is None:
        return 2
    kv_per_token = sum(
        h * (model.head_dim_k + model.head_dim_v) for h in model.n_head_kv_per_layer if h > 0
    ) * 2
    data = {
        "name": model.name,
        "architecture": model.arch,
        "quantization": model.file_type,
        "gguf_version": model.version,
        "tensors": model.tensor_count,
        "layers": model.n_layer,
        "embedding_length": model.n_embd,
        "heads": model.n_head,
        "kv_heads_per_layer": model.n_head_kv_per_layer[:8],
        "head_dim_k": model.head_dim_k,
        "head_dim_v": model.head_dim_v,
        "context_train": model.n_ctx_train,
        "sliding_window": model.sliding_window,
        "vocab": model.n_vocab,
        "moe": model.is_moe,
        "weights_gib": round(model.weights_bytes / GIB, 2),
        "kv_bytes_per_token_f16": kv_per_token,
    }
    if args.json:
        print(json.dumps(data, indent=2))
    else:
        ui.info_rows(data)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="oldiron",
        description="Run local LLMs on the hardware you already have. "
                    "Checks old NVIDIA GPUs, Apple Silicon and CPUs for llama.cpp "
                    "compatibility, and works out what actually fits in memory.",
    )
    parser.add_argument("--version", action="version", version=f"oldiron {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(sub_parser):
        sub_parser.add_argument("--plain", action="store_true",
                                help="plain text output even when rich is installed")
        sub_parser.add_argument("--json", action="store_true")
        return sub_parser

    doctor = common(sub.add_parser("doctor", help="inspect hardware and print a build recipe"))
    doctor.set_defaults(func=cmd_doctor)

    model_help = ("path to a .gguf file, or hf:<user>/<repo>[:QUANT] to read the "
                  "metadata straight off Hugging Face without downloading")
    fit = common(sub.add_parser("fit", help="check whether a GGUF model fits and how to launch it"))
    fit.add_argument("model", help=model_help)
    fit.add_argument("-c", "--ctx", type=int, default=None, help="desired context size")
    fit.add_argument("-ub", "--ubatch", type=int, default=512)
    fit.add_argument("--no-table", action="store_true", help="skip the max-context table")
    fit.add_argument("--hf-token", default=None, help="token for gated repos (or set HF_TOKEN)")
    fit.set_defaults(func=cmd_fit)

    bench_cmd = common(sub.add_parser(
        "bench", help="measure real tokens/s with llama-bench and compare to the ceiling"))
    bench_cmd.add_argument("model", help="path to a .gguf file (must be on disk)")
    bench_cmd.add_argument("-c", "--ctx", type=int, default=None)
    bench_cmd.add_argument("-ub", "--ubatch", type=int, default=512)
    bench_cmd.add_argument("-p", "--prompt-tokens", type=int, default=512,
                           help="prompt length for the prompt-processing test")
    bench_cmd.add_argument("-n", "--gen-tokens", type=int, default=128,
                           help="tokens to generate for the generation test")
    bench_cmd.add_argument("-r", "--repetitions", type=int, default=3)
    bench_cmd.add_argument("--llama-bench", default=None,
                           help="path to the llama-bench binary")
    bench_cmd.add_argument("-o", "--output", default=None,
                           help="where to write the result record (default: results/)")
    bench_cmd.add_argument("--no-save", action="store_true",
                           help="print the result without writing a file")
    bench_cmd.set_defaults(func=cmd_bench)

    info = common(sub.add_parser("info", help="print GGUF metadata relevant to memory use"))
    info.add_argument("model", help=model_help)
    info.add_argument("--hf-token", default=None)
    info.set_defaults(func=cmd_info)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
