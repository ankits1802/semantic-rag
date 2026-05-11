"""
main.py — CLI entry point for the Context-Aware Retrieval Engine.

Usage examples
--------------
# Single query with side-by-side comparison
python main.py query --query "How does the system handle peak load?"

# Run full benchmark suite
python main.py benchmark

# Ingest custom documents and query
python main.py query --query "How does autoscaling work?" --data-dir ./my_docs/

# Use a specific expansion mode
python main.py query --query "rate limiting" --expansion-mode hyde --top-k 3

# Save benchmark results
python main.py benchmark --output ./outputs/my_benchmark.json

# Show index statistics
python main.py info
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich import print as rprint

from src.benchmarking.benchmark_engine import BenchmarkEngine, DEFAULT_QUERY_BANK
from src.retrieval.orchestrator import ContextAwareRetriever
from src.utils.logger import configure_root_logger, get_logger

console = Console()
logger = get_logger(__name__)

_DEFAULT_DATA_DIR = str(
    pathlib.Path(__file__).parent / "data" / "documents"
)


# ── CLI group ─────────────────────────────────────────────────────────────────

@click.group()
@click.option("--config", default=None, help="Path to config.yaml")
@click.option("--log-level", default="INFO", help="Log level: DEBUG|INFO|WARNING|ERROR")
@click.pass_context
def cli(ctx: click.Context, config: str, log_level: str) -> None:
    """Context-Aware Retrieval Engine — Semantic RAG & Vector Search Assessment."""
    configure_root_logger(level=log_level)
    ctx.ensure_object(dict)
    ctx.obj["config"] = config


# ── query command ──────────────────────────────────────────────────────────────

@cli.command()
@click.option("--query", "-q", required=True, help="Search query string.")
@click.option("--top-k", default=5, show_default=True, help="Number of results.")
@click.option(
    "--expansion-mode",
    default="full",
    type=click.Choice(["full", "synonyms", "technical", "hyde"]),
    show_default=True,
    help="Query expansion mode for Strategy B.",
)
@click.option("--data-dir", default=_DEFAULT_DATA_DIR, show_default=True)
@click.option("--no-rerank", is_flag=True, default=False, help="Disable re-ranking.")
@click.option("--json-output", is_flag=True, default=False, help="Output raw JSON.")
@click.pass_context
def query(
    ctx: click.Context,
    query: str,
    top_k: int,
    expansion_mode: str,
    data_dir: str,
    no_rerank: bool,
    json_output: bool,
) -> None:
    """Run a single query through both strategies and compare results."""
    retriever = _build_retriever(ctx.obj["config"], data_dir)

    with Progress(SpinnerColumn(), TextColumn("[bold blue]{task.description}"), console=console) as progress:
        task = progress.add_task("Retrieving...", total=None)
        result_a = retriever.retrieve_raw(query, top_k=top_k)
        result_b = retriever.retrieve_enhanced(query, top_k=top_k, mode=expansion_mode)
        progress.remove_task(task)

    if json_output:
        output = {
            "query": query,
            "strategy_a": result_a.to_dict(),
            "strategy_b": result_b.to_dict(),
        }
        click.echo(json.dumps(output, indent=2))
        return

    # ── Pretty console output ─────────────────────────────────────────────────
    console.print(f"\n[bold yellow]Query:[/bold yellow] {query}\n")

    # Strategy A
    console.print(Panel.fit("[bold cyan]Strategy A — Direct Vector Search[/bold cyan]"))
    _print_results_table(result_a.retrieved_chunks)
    console.print(f"  Latency: {result_a.latency_ms:.1f} ms\n")

    # Strategy B
    exp = result_b.expanded_query
    console.print(Panel.fit("[bold magenta]Strategy B — AI-Enhanced Retrieval[/bold magenta]"))
    console.print(f"  [bold]Expanded:[/bold] {exp.expanded_query[:200]}")
    if exp.keywords_added:
        console.print(f"  [bold]Keywords added:[/bold] {', '.join(exp.keywords_added[:8])}")
    console.print()
    _print_results_table(result_b.retrieved_chunks)
    console.print(f"  Latency: {result_b.latency_ms:.1f} ms\n")


# ── benchmark command ──────────────────────────────────────────────────────────

@cli.command()
@click.option("--data-dir", default=_DEFAULT_DATA_DIR, show_default=True)
@click.option(
    "--output",
    default="./outputs/benchmark_results/benchmark_latest.json",
    show_default=True,
)
@click.option("--top-k", default=5, show_default=True)
@click.option("--json-only", is_flag=True, default=False, help="Skip Markdown report.")
@click.pass_context
def benchmark(
    ctx: click.Context,
    data_dir: str,
    output: str,
    top_k: int,
    json_only: bool,
) -> None:
    """Run the full benchmark suite on all test queries."""
    retriever = _build_retriever(ctx.obj["config"], data_dir)

    console.print("\n[bold]Running benchmark on [cyan]%d queries[/cyan]...[/bold]\n" % len(DEFAULT_QUERY_BANK))

    engine = BenchmarkEngine(retriever, top_k=top_k)

    with Progress(SpinnerColumn(), TextColumn("[bold blue]{task.description}"), console=console) as progress:
        task = progress.add_task("Benchmarking all queries...", total=None)
        report = engine.run()
        progress.remove_task(task)

    # Print aggregate summary
    console.print(Panel.fit("[bold green]Benchmark Complete[/bold green]"))
    _print_aggregate_table(report.aggregate_metrics_a, report.aggregate_metrics_b)
    console.print(f"\n[dim]{report.overall_analysis}[/dim]\n")

    # Save
    saved_path = engine.save_report(
        report,
        output_dir=str(pathlib.Path(output).parent),
    )
    console.print(f"[green]✓[/green] Report saved: {saved_path}")


# ── info command ───────────────────────────────────────────────────────────────

@cli.command()
@click.option("--data-dir", default=_DEFAULT_DATA_DIR, show_default=True)
@click.pass_context
def info(ctx: click.Context, data_dir: str) -> None:
    """Show index and pipeline statistics."""
    retriever = _build_retriever(ctx.obj["config"], data_dir)
    telemetry = retriever.embedding_service.telemetry()

    table = Table(title="Pipeline Statistics")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("Documents indexed", str(retriever.num_chunks) + " chunks")
    table.add_row("Embedding model", telemetry["model"])
    table.add_row("Embedding dimension", str(telemetry["dimension"]))
    table.add_row("Cache entries", str(telemetry["cache_entries"]))
    table.add_row("Cache hit rate", f"{telemetry['cache_hit_rate'] * 100:.1f}%")
    console.print(table)


# ── helpers ───────────────────────────────────────────────────────────────────

def _build_retriever(config_path: str | None, data_dir: str) -> ContextAwareRetriever:
    """Initialise and warm up the retriever pipeline."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Loading pipeline...", total=None)
        retriever = ContextAwareRetriever(config_path=config_path)
        retriever.setup(data_dir=data_dir)
        progress.remove_task(task)
    return retriever


def _print_results_table(results: list) -> None:
    table = Table(show_header=True, header_style="bold")
    table.add_column("Rank", width=5)
    table.add_column("Score", width=7)
    table.add_column("Source", width=25)
    table.add_column("Section", width=20)
    table.add_column("Text (truncated)", width=70)
    for r in results:
        table.add_row(
            str(r.rank),
            f"{r.score:.4f}",
            r.source[:24],
            getattr(r, "section", "")[:18],
            r.text[:120].replace("\n", " "),
        )
    console.print(table)


def _print_aggregate_table(
    agg_a: dict,
    agg_b: dict,
) -> None:
    table = Table(title="Aggregate Metrics (mean across all queries)")
    table.add_column("Metric", style="cyan")
    table.add_column("Strategy A", style="white")
    table.add_column("Strategy B", style="white")
    table.add_column("Δ (B - A)", style="green")
    for key in sorted(set(agg_a) | set(agg_b)):
        va = agg_a.get(key, 0)
        vb = agg_b.get(key, 0)
        delta = vb - va
        delta_str = f"{delta:+.4f}"
        table.add_row(key, f"{va:.4f}", f"{vb:.4f}", delta_str)
    console.print(table)


if __name__ == "__main__":
    cli()
