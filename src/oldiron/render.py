"""Output rendering.

Two renderers behind one interface: a plain one that needs nothing but the
standard library, and a rich one used when `rich` is installed. The plain
renderer is the reference -- oldiron has to stay installable on the machine it
is diagnosing, so pretty output is an extra, never a requirement.
"""

from __future__ import annotations

from .advisor import ERROR, OK, WARN, Finding, RunPlan
from .gguf_meta import GgufModel
from .hardware import System
from .memory import Budget, human

try:  # pragma: no cover - exercised by whichever environment runs the tests
    from rich.box import ROUNDED
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.table import Table
    from rich.text import Text

    HAS_RICH = True
except ImportError:  # pragma: no cover
    HAS_RICH = False

MARKS = {OK: "[ ok ]", WARN: "[warn]", ERROR: "[FAIL]"}
ICONS = {OK: "✓", WARN: "!", ERROR: "✗"}
STYLES = {OK: "green", WARN: "yellow", ERROR: "bold red"}

# Budget segment -> colour, shared by the bar and the table labels.
SEGMENTS = (
    ("weights", "cyan"),
    ("KV cache", "magenta"),
    ("compute buffers", "blue"),
    ("runtime overhead", "yellow"),
)


def _segment_values(budget: Budget) -> list[int]:
    return [budget.weights, budget.kv_cache, budget.compute, budget.overhead]


class PlainRenderer:
    """Exactly what oldiron printed before rich existed."""

    def __init__(self, write=print):
        self._write = write

    def out(self, text: str = "") -> None:
        self._write(text)

    # --- shared blocks -----------------------------------------------------
    def system(self, system: System) -> None:
        self.out(f"System   {system.os}")
        self.out(f"CPU      {system.cpu.model.strip() or 'unknown'} "
                 f"({system.cpu.cores} threads, SIMD: {system.cpu.simd_level})")
        self.out(f"RAM      {human(system.ram_bytes)}" if system.ram_bytes
                 else "RAM      unknown")
        for gpu in system.gpus:
            self.out(f"GPU {gpu.index}    {_gpu_line(gpu)}")
        if not system.gpus:
            self.out("GPU      none detected")
        if system.backend == "metal":
            self.out("Backend  Metal (unified memory)")
        else:
            self.out(f"Driver   {system.driver_version or 'unknown'}        "
                     f"CUDA: {system.cuda_version or 'not found'}")

    def findings(self, findings: list[Finding]) -> None:
        if not findings:
            return
        self.out()
        for f in findings:
            self.out(f"{MARKS[f.level]} {f.title}")
            if f.detail:
                self.out(f"       {f.detail}")
            if f.fix:
                self.out(f"       fix: {f.fix}")

    def build_recipe(self, lines: list[str]) -> None:
        self.out("\nBuild recipe for this machine:\n")
        for line in lines:
            self.out(f"  {line}")

    def model(self, model: GgufModel) -> None:
        self.out(f"Model    {model.name}  [{model.arch}, {model.file_type}]")
        self.out(f"         {model.n_layer} layers, trained context "
                 f"{model.n_ctx_train or '?'}, {len(model.shard_paths)} file(s)")
        if model.remote_ref:
            self.out(f"         {model.remote_ref} on Hugging Face "
                     "(metadata only, not downloaded)")

    def plan(self, plan: RunPlan, system: System) -> None:
        budget = plan.budget
        metal = system.backend == "metal"
        overhead_label = "runtime overhead" if metal else "CUDA overhead"
        available_label = "GPU memory cap " if metal else "available VRAM"
        self.out()
        self.out(f"Plan     context {plan.ctx}, KV cache {plan.ctk}/{plan.ctv}, "
                 f"-ngl {plan.n_gpu_layers}")
        self.out()
        weights_label = ("weights on GPU" if budget.partial else "weights")
        self.out(f"  {weights_label:<16} {human(budget.weights):>12}")
        self.out(f"  KV cache         {human(budget.kv_cache):>12}")
        self.out(f"  compute buffers  {human(budget.compute):>12}  (estimate)")
        self.out(f"  {overhead_label:<16} {human(budget.overhead):>12}  (estimate)")
        self.out(f"  {'-' * 32}")
        self.out(f"  total            {human(budget.total):>12}")
        self.out(f"  {available_label:<16} {human(budget.available):>12}")
        if budget.fits:
            self.out(f"  headroom         {human(budget.headroom):>12}  (fits)")
        else:
            self.out(f"  short by         {human(-budget.headroom):>12}  (DOES NOT FIT)")
        if budget.partial:
            self.out(f"  on CPU           {human(budget.weights_cpu):>12}  (host RAM)")
        if plan.decode_tps and not budget.partial:
            self.out(f"\nSpeed    ceiling ~{plan.decode_tps:.1f} tok/s "
                     "(memory bandwidth; real throughput is lower)")
        elif plan.decode_tps:
            self.out(f"\nSpeed    the ~{plan.decode_tps:.1f} tok/s bandwidth ceiling does not "
                     "apply with layers on the CPU; expect far less")

    def context_table(self, rows: list[tuple[str, int]]) -> None:
        self.out("\nMax context by KV cache type (fully offloaded):")
        for label, ctx in rows:
            self.out(f"  {label:<14} {ctx:>8} tokens"
                     + ("   (does not fit)" if ctx == 0 else ""))

    def notes(self, notes: list[str]) -> None:
        for note in notes:
            self.out(f"\nnote: {note}")

    def command(self, command: str) -> None:
        self.out("\nStart with:\n")
        self.out(f"  {command}")

    def footer(self, text: str) -> None:
        self.out(f"\n{text}")

    def info_rows(self, data: dict) -> None:
        for key, value in data.items():
            self.out(f"{key:<24} {value}")

    def bench(self, result, saved_to=None) -> None:
        self.out("\nMeasured:")
        for label, run in (("prompt processing", result.prompt),
                           ("generation", result.generation)):
            if run:
                self.out(f"  {label:<20} {run.avg_ts:>8.2f} tok/s"
                         + (f"  +-{run.stddev_ts:.2f}" if run.stddev_ts else ""))
        if result.ceiling_tps and not result.partial_offload:
            self.out(f"  {'bandwidth ceiling':<20} {result.ceiling_tps:>8.2f} tok/s")
        if result.efficiency is not None:
            self.out(f"  {'efficiency':<20} {result.efficiency * 100:>7.1f}% of ceiling")
        elif result.partial_offload:
            self.out("\nPartial offload: some layers ran on the CPU, so this does not "
                     "measure the card.\nRe-run at a context that fits entirely on the "
                     "GPU for a comparable number.")
        if result.build_commit:
            self.out(f"\nllama.cpp {result.build_commit}")
        if saved_to:
            self.out(f"\nResult written to {saved_to}")
            self.out("Send it in as a pull request to add this card to the table.")

    def error(self, message: str) -> None:
        self.out(f"error: {message}")


def _gpu_line(gpu) -> str:
    if gpu.compute_cap:
        arch = f"{gpu.arch}, cc {gpu.compute_cap}" if gpu.arch else f"cc {gpu.compute_cap}"
    else:
        arch = gpu.arch or "unknown"
    if gpu.cores:
        arch += f", {gpu.cores}-core GPU"
    label = "shared" if gpu.unified else "VRAM"
    line = f"{gpu.name} -- {human(gpu.vram_bytes)} {label} ({arch})"
    if gpu.bandwidth_gbs:
        line += f"  {gpu.bandwidth_gbs:.0f} GB/s"
    return line


class RichRenderer:
    """Same information, laid out with panels, tables and a memory bar."""

    def __init__(self, console=None):
        self.console = console or Console()

    def out(self, text: str = "") -> None:
        self.console.print(text)

    # --- shared blocks -----------------------------------------------------
    def _hardware_grid(self, system: System) -> Table:
        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="dim", justify="right", no_wrap=True)
        grid.add_column()
        # Values go through Text, never markup strings: a model or device name
        # containing [brackets] would otherwise be parsed as rich markup.
        grid.add_row("os", Text(system.os))
        cpu = system.cpu
        grid.add_row("cpu", Text.assemble(
            (cpu.model.strip() or "unknown", ""),
            (f"  ({cpu.cores} threads, {cpu.simd_level})", "dim"),
        ))
        grid.add_row("ram", Text(human(system.ram_bytes) if system.ram_bytes else "unknown"))
        for gpu in system.gpus:
            name = Text(gpu.name, style="bold")
            detail = Text()
            detail.append(f"{human(gpu.vram_bytes)} ", style="bold green")
            detail.append("shared " if gpu.unified else "VRAM ", style="dim")
            meta = gpu.arch or "unknown"
            if gpu.compute_cap:
                meta += f", cc {gpu.compute_cap}"
            if gpu.cores:
                meta += f", {gpu.cores}-core GPU"
            if gpu.bandwidth_gbs:
                meta += f", {gpu.bandwidth_gbs:.0f} GB/s"
            detail.append(f"({meta})", style="dim")
            grid.add_row(f"gpu {gpu.index}", Text.assemble(name, "  ", detail))
        if not system.gpus:
            grid.add_row("gpu", Text("none detected", style="yellow"))
        if system.backend == "metal":
            grid.add_row("backend", Text.assemble(
                ("Metal", ""), ("  (unified memory)", "dim")))
        else:
            grid.add_row("driver", Text.assemble(
                (system.driver_version or "unknown", ""),
                (f"   cuda: {system.cuda_version or 'not found'}", "dim"),
            ))
        return grid

    def system(self, system: System) -> None:
        self.console.print(Panel(self._hardware_grid(system), title="hardware",
                                 title_align="left", box=ROUNDED, border_style="dim",
                                 padding=(0, 1)))

    def findings(self, findings: list[Finding]) -> None:
        if not findings:
            return
        for f in findings:
            body = Text()
            if f.detail:
                body.append(f.detail)
            if f.fix:
                if f.detail:
                    body.append("\n")
                body.append("fix: ", style="bold")
                body.append(f.fix)
            title = Text.assemble(
                (f" {ICONS[f.level]} ", STYLES[f.level]), (f.title, "bold")
            )
            self.console.print(Panel(body or Text(""), title=title, title_align="left",
                                     border_style=STYLES[f.level], box=ROUNDED,
                                     padding=(0, 1)))

    def build_recipe(self, lines: list[str]) -> None:
        self.console.print(Panel(
            Syntax("\n".join(lines), "bash", theme="ansi_dark", word_wrap=True),
            title="build recipe for this machine", title_align="left",
            border_style="cyan", box=ROUNDED, padding=(0, 1),
        ))

    def model(self, model: GgufModel) -> None:
        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="dim", justify="right", no_wrap=True)
        grid.add_column()
        grid.add_row("name", Text(model.name, style="bold"))
        grid.add_row("format", Text.assemble(
            (model.arch, ""), ("  ", ""), (model.file_type, "bold")))
        grid.add_row("shape", Text(
            f"{model.n_layer} layers, trained context {model.n_ctx_train or '?'}, "
            f"{len(model.shard_paths)} file(s)"))
        if model.remote_ref:
            grid.add_row("source", Text.assemble(
                (model.remote_ref, "bold"),
                (" on Hugging Face (metadata only, not downloaded)", "dim"),
            ))
        self.console.print(Panel(grid, title="model", title_align="left",
                                 border_style="dim", box=ROUNDED, padding=(0, 1)))

    def _budget_bar(self, budget: Budget, width: int = 46) -> Text:
        """One line showing how the available memory is spent."""
        available = max(budget.available, 1)
        scale = max(budget.total, available)
        bar = Text()
        drawn = 0
        for (label, colour), value in zip(SEGMENTS, _segment_values(budget)):
            cells = round(value / scale * width)
            if value > 0 and cells == 0:
                cells = 1
            bar.append("█" * cells, style=colour)
            drawn += cells
        if budget.fits:
            bar.append("░" * max(width - drawn, 0), style="dim")
        else:
            over = round(-budget.headroom / scale * width)
            bar.append("▓" * max(over, 1), style="bold red")
        return bar

    def plan(self, plan: RunPlan, system: System) -> None:
        budget = plan.budget
        metal = system.backend == "metal"
        labels = list(SEGMENTS)
        if not metal:
            labels[3] = ("CUDA overhead", "yellow")

        if budget.partial:
            labels[0] = ("weights on GPU", "cyan")

        table = Table.grid(padding=(0, 2))
        table.add_column(width=18)
        table.add_column(justify="right", width=11)
        table.add_column(style="dim")
        for (label, colour), value in zip(labels, _segment_values(budget)):
            note = "estimate" if label.startswith(("compute", "CUDA", "runtime")) else ""
            table.add_row(Text(label, style=colour), human(value), note)
        table.add_row(Text("total", style="bold"), Text(human(budget.total), style="bold"), "")
        table.add_row(
            Text("GPU memory cap" if metal else "available VRAM", style="dim"),
            human(budget.available), "",
        )
        # Value and qualifier live in separate columns, or the qualifier wraps.
        if budget.fits:
            table.add_row(Text("headroom"),
                          Text(human(budget.headroom), style="green"), "free")
        else:
            table.add_row(Text("over budget"),
                          Text(human(-budget.headroom), style="bold red"),
                          Text("DOES NOT FIT", style="bold red"))
        if budget.partial:
            table.add_row(Text("on CPU", style="dim"),
                          Text(human(budget.weights_cpu), style="yellow"), "host RAM")

        header = Text.assemble(
            ("context ", "dim"), (f"{plan.ctx:,}", "bold"),
            ("   kv cache ", "dim"), (f"{plan.ctk}/{plan.ctv}", "bold"),
            ("   -ngl ", "dim"), (str(plan.n_gpu_layers), "bold"),
        )
        blocks = [header, "", self._budget_bar(budget), "", table]
        if plan.decode_tps and not budget.partial:
            blocks += ["", Text.assemble(
                ("speed  ", "dim"), (f"ceiling ~{plan.decode_tps:.1f} tok/s", "bold"),
                ("  memory bandwidth; real throughput is lower", "dim"),
            )]
        elif plan.decode_tps:
            # The bandwidth ceiling assumes every weight sits in GPU memory.
            blocks += ["", Text.assemble(
                ("speed  ", "dim"), ("bandwidth ceiling does not apply", "bold"),
                (f"  ({plan.decode_tps:.1f} tok/s fully offloaded); expect far less "
                 "with layers on the CPU", "dim"),
            )]

        if not budget.fits:
            border, title = "red", "plan — does not fit"
        elif budget.partial:
            border, title = "yellow", "plan — partial offload"
        else:
            border, title = "green", "plan"
        self.console.print(Panel(Group(*blocks), title=title, title_align="left",
                                 border_style=border, box=ROUNDED, padding=(1, 2)))

    def context_table(self, rows: list[tuple[str, int]]) -> None:
        # The heading goes outside the table: inside, it wraps to the table's
        # own narrow width instead of the terminal's.
        self.console.print(Text("max context, fully offloaded", style="dim"))
        table = Table(box=ROUNDED, border_style="dim", header_style="dim",
                      padding=(0, 1))
        table.add_column("kv cache")
        table.add_column("tokens", justify="right")
        best = max((ctx for _, ctx in rows), default=0)
        for label, ctx in rows:
            if ctx == 0:
                table.add_row(Text(label, style="dim"), Text("does not fit", style="red"))
            else:
                style = "bold green" if ctx == best and best else ""
                table.add_row(label, Text(f"{ctx:,}", style=style))
        self.console.print(table)

    def notes(self, notes: list[str]) -> None:
        for note in notes:
            self.console.print(Text.assemble(("note  ", "bold yellow"), (note, "")))

    def command(self, command: str) -> None:
        self.console.print(Panel(
            Syntax(command, "bash", theme="ansi_dark", word_wrap=True),
            title="start with", title_align="left", border_style="green",
            box=ROUNDED, padding=(0, 1),
        ))

    def footer(self, text: str) -> None:
        self.console.print(Text(text, style="dim"))

    def info_rows(self, data: dict) -> None:
        table = Table(box=ROUNDED, border_style="dim", show_header=False, padding=(0, 1))
        table.add_column(style="dim", no_wrap=True)
        table.add_column()
        for key, value in data.items():
            table.add_row(key, str(value))
        self.console.print(table)

    def bench(self, result, saved_to=None) -> None:
        # Units live in their own column: inside the value column they wrap.
        table = Table.grid(padding=(0, 2))
        table.add_column(width=18)
        table.add_column(justify="right", width=8)
        table.add_column(style="dim")
        if result.prompt:
            table.add_row(Text("prompt processing", style="blue"),
                          Text(f"{result.prompt.avg_ts:.2f}", style="bold"),
                          "tok/s" + (f"  +-{result.prompt.stddev_ts:.2f}"
                                     if result.prompt.stddev_ts else ""))
        if result.generation:
            table.add_row(Text("generation", style="cyan"),
                          Text(f"{result.generation.avg_ts:.2f}", style="bold"),
                          "tok/s" + (f"  +-{result.generation.stddev_ts:.2f}"
                                     if result.generation.stddev_ts else ""))
        if result.ceiling_tps and not result.partial_offload:
            table.add_row(Text("bandwidth ceiling", style="dim"),
                          Text(f"{result.ceiling_tps:.2f}"), "tok/s  theoretical")

        blocks = [table]
        if result.partial_offload:
            blocks += ["", Text.assemble(
                ("partial offload", "bold yellow"),
                (" — layers ran on the CPU, so this measures the pair, not the card.\n"
                 "Re-run at a context that fits entirely on the GPU for a number that "
                 "compares with other cards.", "dim"),
            )]
        if result.efficiency is not None:
            share = result.efficiency
            colour = "green" if share >= 0.6 else "yellow" if share >= 0.35 else "red"
            width = 34
            filled = max(1, min(width, round(share * width)))
            bar = Text("█" * filled, style=colour)
            bar.append("░" * (width - filled), style="dim")
            blocks += ["", Text.assemble(
                (f"{share * 100:.0f}% of the ceiling", f"bold {colour}"),
                ("  — what this card actually delivers", "dim"),
            ), bar]
        if result.build_commit:
            blocks += ["", Text(f"llama.cpp {result.build_commit}", style="dim")]

        self.console.print(Panel(Group(*blocks), title="measured", title_align="left",
                                 border_style="cyan", box=ROUNDED, padding=(1, 2)))
        if saved_to:
            self.console.print(Text.assemble(
                ("saved  ", "dim"), (str(saved_to), "bold"),
                ("\n       send it in as a pull request to add this card to the table",
                 "dim"),
            ))

    def error(self, message: str) -> None:
        self.console.print(Text.assemble(("error  ", "bold red"), (message, "")))


def get_renderer(plain: bool = False, console=None):
    """Rich when it is installed and wanted, plain otherwise."""
    if plain or not HAS_RICH:
        return PlainRenderer()
    return RichRenderer(console)
