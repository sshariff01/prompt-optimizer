"""CLI entry point for the prompt optimizer."""

import json
import sys
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

# Import tomllib for Python 3.11+, tomli for earlier versions
try:
    import tomllib
except ImportError:
    import tomli as tomllib

from src.optimizer.models import EvalCase, OptimizationConfig
from src.orchestrator.optimization_loop import OptimizationLoop
from src.providers.anthropic import AnthropicProvider

# Load .env file if it exists
load_dotenv()

app = typer.Typer(help="Prompt Optimizer - Iterative meta-prompt refinement system")
console = Console()


@app.command()
def optimize(
    config_path: Path = typer.Argument(
        ...,
        help="Path to configuration TOML file",
        exists=True,
    ),
):
    """Optimize a prompt based on configuration and training data."""
    try:
        # Check for API key
        import os
        if not os.environ.get("ANTHROPIC_API_KEY"):
            console.print("\n[bold red]Error: ANTHROPIC_API_KEY not found[/bold red]")
            console.print("\nPlease set your API key using one of these methods:")
            console.print("1. Create a .env file: cp .env.example .env")
            console.print("   Then edit .env and add: ANTHROPIC_API_KEY=your_key_here")
            console.print("2. Export as environment variable:")
            console.print("   export ANTHROPIC_API_KEY=your_key_here\n")
            raise typer.Exit(code=1)

        # Load configuration
        console.print("\n[bold blue]Loading configuration...[/bold blue]")
        with open(config_path, "rb") as f:
            config_dict = tomllib.load(f)

        config = OptimizationConfig(**config_dict)
        console.print(f"✓ Configuration loaded from {config_path}\n")

        # Load training data
        console.print("[bold blue]Loading training data...[/bold blue]")
        training_cases = load_jsonl(config.data.training_set)
        console.print(f"✓ Loaded {len(training_cases)} training cases")

        # Load test data
        test_cases = load_jsonl(config.data.test_set)
        console.print(f"✓ Loaded {len(test_cases)} test cases\n")

        # Initialize providers
        console.print("[bold blue]Initializing providers...[/bold blue]")

        # Optimizer provider (Opus for meta-optimization)
        optimizer_provider = AnthropicProvider(model=config.optimizer.model)
        console.print(f"✓ Optimizer: {config.optimizer.model}")

        # Target provider (model being optimized)
        target_provider = AnthropicProvider(
            model=config.target_model.model,
        )
        console.print(f"✓ Target model: {config.target_model.model}\n")

        # Create optimization loop
        optimization_loop = OptimizationLoop(
            config=config,
            optimizer_provider=optimizer_provider,
            target_provider=target_provider,
        )

        # Run optimization
        console.print("[bold green]Starting optimization...[/bold green]\n")
        console.print("=" * 70)
        console.print()

        result = optimization_loop.optimize(training_cases, test_cases)

        # Display results
        console.print("=" * 70)
        console.print()
        console.print("[bold green]Optimization Complete![/bold green]\n")

        # Status panel
        status_color = "green" if result.status.value == "success" else "yellow"
        console.print(
            Panel(
                f"[bold]{result.message}[/bold]",
                title=f"Status: {result.status.value}",
                border_style=status_color,
            )
        )

        # Metrics
        console.print("\n[bold]Metrics:[/bold]")
        console.print(f"  Training Pass Rate: {result.training_pass_rate:.1%}")
        console.print(f"  Test Pass Rate: {result.test_pass_rate:.1%}")
        console.print(f"  Total Iterations: {len(result.iterations)}")
        console.print(f"  Optimizer Tokens Used: {result.total_optimizer_tokens:,}")

        # Adaptive optimization stats
        stats = optimization_loop.prompt_history.get_stats()
        console.print(f"\n[bold]Adaptive Optimization Stats:[/bold]")
        console.print(f"  Refinements Accepted: {stats['accepted']}")
        console.print(f"  Refinements Rejected: {stats['rejected']}")
        console.print(f"  Acceptance Rate: {stats['acceptance_rate']:.1%}")
        console.print()

        # Final prompt
        console.print("[bold]Final Optimized Prompt:[/bold]")
        syntax = Syntax(result.final_prompt, "text", theme="monokai", word_wrap=True)
        console.print(Panel(syntax, border_style="blue"))

        # Save result
        output_path = config_path.parent / "optimization_result.json"
        save_result(result, output_path)
        console.print(f"\n✓ Results saved to {output_path}\n")

    except Exception as e:
        console.print(f"\n[bold red]Error: {e}[/bold red]\n")
        raise typer.Exit(code=1)


def load_jsonl(path: Path) -> list[EvalCase]:
    """Load evaluation cases from JSONL file.

    Args:
        path: Path to JSONL file

    Returns:
        List of evaluation cases
    """
    cases = []

    with open(path, "r") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                cases.append(
                    EvalCase(
                        input=data["input"],
                        expected_output=data["expected_output"],
                    )
                )

    return cases


def save_result(result, path: Path):
    """Save optimization result to JSON file.

    Args:
        result: OptimizationResult object
        path: Output path
    """
    # Convert to dict for JSON serialization
    result_dict = result.model_dump()

    with open(path, "w") as f:
        json.dump(result_dict, f, indent=2)


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
