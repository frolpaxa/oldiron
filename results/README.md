# Benchmark results

What old hardware actually delivers, as measured by `oldiron bench`.

A GPU's memory bandwidth sets a hard ceiling on generation speed, but how close a real
run gets depends on the architecture, the backend and the build. Recent cards land near
that ceiling; older ones, without tensor cores and with kernels tuned elsewhere, fall
much further short. Nobody publishes those numbers, so this directory collects them.

## Adding yours

```bash
oldiron bench ~/models/your-model.gguf -c 16384
```

The command runs `llama-bench` with the settings `oldiron fit` recommends, prints what it
measured, and writes a JSON record into this directory. Open a pull request with that
file — one file per card, model and quant combination.

Nothing about you is recorded: the file holds hardware, model, flags and timings, and you
can read it before sending it. `--no-save` skips writing it altogether.

## What makes a comparable number

- **Run a configuration that fits entirely on the GPU.** With layers on the CPU the
  measurement describes the card and the host together, and the bandwidth ceiling stops
  applying, so `oldiron` withholds the efficiency figure. `fit` will tell you which
  context fits.
- Let the machine be idle otherwise. A desktop session or another model still loaded will
  quietly halve the result.
- Keep the defaults for `-p` and `-n` unless you have a reason not to, so runs stay
  comparable.
- Say so in the pull request if the card is power-limited or thermally throttled, or if
  the build differs from the recipe `oldiron doctor` prints.
