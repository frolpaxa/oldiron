# Command line

Three commands. Every one of them accepts `--json` for scripting and `--plain` to force
the dependency-free text layout even when `rich` is installed.

## `oldiron doctor`

Inspects the machine and prints a build recipe tailored to it.

```bash
oldiron doctor
oldiron doctor --json
```

It reports the CPU and its SIMD level, RAM, every detected GPU with its architecture and
memory bandwidth, the driver and CUDA toolkit versions, and then a list of findings:

| Mark | Meaning |
| --- | --- |
| `ok` | Worth knowing, nothing to do. |
| `warn` | Will cost performance, or will surprise you later. |
| `FAIL` | This combination cannot work as it stands. |

The exit status is `1` if any finding is a failure, otherwise `0`, so it can gate a
setup script.

## `oldiron fit`

Answers whether a model fits, and prints the command to launch it.

```bash
oldiron fit MODEL [-c CTX] [-ub UBATCH] [--no-table] [--hf-token TOKEN]
```

`MODEL` is a path to a `.gguf` file, or a Hugging Face reference (see
[Checking before you download](#checking-before-you-download) below).

| Option | Meaning |
| --- | --- |
| `-c`, `--ctx` | Context size you want. Without it, oldiron picks the largest sensible one that fits. |
| `-ub`, `--ubatch` | Physical batch size, matching llama.cpp's `-ub`. Affects the compute buffer estimate. Default 512. |
| `--no-table` | Skip the "max context by KV cache type" table. |
| `--hf-token` | Token for gated repositories. `HF_TOKEN` in the environment works too. |

The output breaks memory into weights, KV cache, compute buffers and runtime overhead,
compares the total against what the GPU can actually allocate, and ends with a ready
`llama-server` command.

If the whole model will not fit, oldiron lowers the KV cache precision first, then reduces
the number of offloaded layers. The budget always describes the plan being recommended: in
a partial offload only the offloaded weights count against VRAM, and the remainder is
reported separately as host RAM. It will also tell you which shorter context would fit
entirely on the GPU, which is usually faster than spilling layers to the CPU.

Exit status is `0` when the printed command will run — including a partial offload — and
`1` when nothing fits at all. Use `partial_offload` in the `--json` output to tell the two
apart.

## `oldiron bench`

Measures what the hardware actually delivers and compares it to the ceiling.

```bash
oldiron bench MODEL [-c CTX] [-p PROMPT_TOKENS] [-n GEN_TOKENS] [-r REPETITIONS]
              [--llama-bench PATH] [-o FILE] [--no-save]
```

It locates `llama-bench` (on `PATH`, in the usual build directories, or wherever
`--llama-bench` points), runs it with the settings `fit` recommends rather than
llama-bench's own defaults, and reports prompt-processing and generation speed next to the
bandwidth ceiling:

```
  prompt processing     465.13  tok/s  +-5.02
  generation             11.42  tok/s  +-0.18
  bandwidth ceiling      55.70  tok/s  theoretical

  21% of the ceiling  — what this card actually delivers
```

That percentage is the point. The ceiling is arithmetic; how close a card gets to it is
an empirical fact that varies by architecture and backend, and for old cards it is
largely unpublished.

Under a partial offload the percentage is withheld rather than shown. The ceiling assumes
every weight is in GPU memory, so with layers on the CPU the ratio would describe the host
processor as much as the card, and would not compare with anyone else's run. The raw
timings are still reported. Re-run at a context that fits entirely on the GPU — `fit` says
which one — for a comparable number.

| Option | Meaning |
| --- | --- |
| `-p`, `--prompt-tokens` | Prompt length for the prompt-processing test. Default 512. |
| `-n`, `--gen-tokens` | Tokens to generate for the generation test. Default 128. |
| `-r`, `--repetitions` | Repetitions per test. Default 3. |
| `--llama-bench` | Path to the binary, when it is not on `PATH`. |
| `-o`, `--output` | Where to write the result record. Default `results/<card>-<model>-<quant>.json`. |
| `--no-save` | Print the measurement without writing a file. |

The record it writes describes the hardware, the model, the flags and the timings, and is
meant to be sent in as a pull request — see `results/README.md` in the repository. The
model has to be on disk: an `hf:` reference cannot be benchmarked without downloading it.

## `oldiron info`

Prints the metadata the arithmetic is based on: layer count, head counts per layer, head
dimensions, trained context, vocabulary size, and the KV cache cost per token.

```bash
oldiron info ~/models/Qwen3-8B-Q4_K_M.gguf
oldiron info hf:unsloth/Qwen3-8B-GGUF:Q4_K_M --json
```

Use it when a `fit` result looks wrong: it shows exactly which numbers came out of the
file.

## Checking before you download

A GGUF file keeps its metadata at the front, so a couple of HTTP range requests are
enough to answer "will this fit?" for a model you have not downloaded.

```bash
# quant defaults to Q4_K_M, as in llama.cpp
oldiron fit hf:unsloth/Qwen3-4B-GGUF

# a specific quant
oldiron fit hf:unsloth/Qwen3-4B-GGUF:Q8_0 -c 16384

# a specific revision
oldiron fit hf:unsloth/Qwen3-4B-GGUF:Q4_K_M@refs/pr/3

# a URL copied from the Hub
oldiron fit https://huggingface.co/unsloth/Qwen3-4B-GGUF/blob/main/Qwen3-4B-Q4_K_M.gguf
```

Naming a quant that does not exist lists the ones that do. Sharded models are summed
across every part, so the weight total is the real one, and the printed command uses
`-hf`, which lets `llama-server` do the download itself.

## Output modes

```bash
oldiron fit model.gguf            # rich layout when rich is installed
oldiron fit model.gguf --plain    # plain text, always available
oldiron fit model.gguf --json     # machine-readable, unaffected by the above
```

The JSON shape is stable enough to script against: `model`, `plan` (including
`fits`, `decode_tps_ceiling` and the byte-level budget), `command` and `notes`.
