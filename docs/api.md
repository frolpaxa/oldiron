# API reference

`oldiron` is a library as well as a command. Everything below is importable.

## Hardware detection

```{eval-rst}
.. automodule:: oldiron.hardware
   :members: System, Cpu, Gpu, detect, parse_nvidia_smi, parse_cpuinfo,
             parse_darwin_x86_features, implied_arm_features, detect_cpu_darwin,
             apple_gpu
```

## GPU database

```{eval-rst}
.. automodule:: oldiron.gpudb
   :members: ArchInfo, AppleChip, arch_for, apple_chip_for, is_legacy
```

## GGUF metadata

```{eval-rst}
.. automodule:: oldiron.gguf_meta
   :members: GgufModel, GgufError, read_gguf, parse_header
```

## Reading models over HTTP

```{eval-rst}
.. automodule:: oldiron.remote
   :members: HfRef, RemoteFile, RemoteError, RangeStream, read_gguf_remote,
             parse_spec, is_remote_spec, parse_tree, pick_file, shard_group,
             resolve_url, token_from_env
```

## Memory arithmetic

```{eval-rst}
.. automodule:: oldiron.memory
   :members: Budget, kv_cache_bytes, compute_buffer_bytes, runtime_overhead,
             plan, max_context, layers_that_fit, decode_ceiling_tps, human
```

## Benchmarking

```{eval-rst}
.. automodule:: oldiron.bench
   :members: Run, BenchResult, BenchError, find_binary, build_command, parse_output,
             measure, submission, suggested_filename, write_submission
```

## Advice

```{eval-rst}
.. automodule:: oldiron.advisor
   :members: Finding, RunPlan, check_system, build_recipe, plan_run
```

## Rendering

```{eval-rst}
.. automodule:: oldiron.render
   :members: PlainRenderer, RichRenderer, get_renderer
```

## Example

```python
from oldiron.advisor import check_system, plan_run
from oldiron.gguf_meta import read_gguf
from oldiron.hardware import detect

system = detect()
problems = [f for f in check_system(system) if f.level == "error"]

model = read_gguf("/models/Qwen3-8B-Q4_K_M.gguf")
plan = plan_run(model, system, ctx=16384)

if plan.budget.fits:
    print(plan.command(model.launch_arg))
else:
    print(f"short by {plan.budget.headroom / 2**30:.2f} GiB")
```
