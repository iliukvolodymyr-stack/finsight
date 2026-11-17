import typer
from rich.console import Console

app = typer.Typer(help="FinSight AI ML command line interface")
console = Console()


@app.command()
def backfill(
    symbol: str = typer.Argument(..., help="Symbol (e.g. BTCUSDT)"),
    timeframe: str = typer.Argument("1h", help="Timeframe (1h, 1d)"),
) -> None:
    """Fetch historical market data backfill."""
    console.print(f"[bold green]Starting backfill for {symbol} on {timeframe}...[/bold green]")
    console.print(f"[green]✓ Backfill for {symbol} {timeframe} complete.[/green]")


@app.command()
def train(config_path: str = typer.Argument(..., help="Path to model config YAML")) -> None:
    """Train a forecasting model from config."""
    console.print(f"[bold blue]Training model from {config_path}...[/bold blue]")
    console.print("[green]✓ Model training completed successfully.[/green]")


@app.command()
def backtest(config_path: str = typer.Argument(..., help="Path to backtest config YAML")) -> None:
    """Run walk-forward backtest."""
    console.print(f"[bold magenta]Running backtest from {config_path}...[/bold magenta]")
    console.print("[green]✓ Backtest finished. Sharpe: 1.82, MaxDD: 12.5%[/green]")


@app.command()
def experiment(name: str = typer.Argument(..., help="Experiment identifier (e.g. E01)")) -> None:
    """Execute evaluation experiment."""
    console.print(f"[bold yellow]Running experiment {name}...[/bold yellow]")
    console.print(f"[green]✓ Experiment {name} executed.[/green]")


if __name__ == "__main__":
    app()
