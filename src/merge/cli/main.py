from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from tabulate import tabulate

from ..runner import sweep
from ..viz.charts import (
    plot_cosine_scatter,
    plot_drift_bar,
    plot_layer_drift_heatmap,
    plot_method_comparison,
    plot_top_layers_grouped,
)

app = typer.Typer(add_completion=False, help="merge: model merging lab")


@app.command("sweep")
def cmd_sweep(
    methods: Annotated[
        str, typer.Option(help="comma-separated methods")
    ] = "linear,slerp,task_arith,ties,dare,model_stock",
    out_dir: Annotated[Path, typer.Option(help="results dir")] = Path("results"),
) -> None:
    method_list = [m.strip() for m in methods.split(",") if m.strip()]
    results = sweep(method_list, out_dir)
    rows = [
        (m, r["cosine_to_base"], r["cosine_to_parent_avg"], r["mean_abs_drift"])
        for m, r in results.items()
    ]
    print()
    print(
        tabulate(
            rows,
            headers=["method", "cos(merged,base)", "cos(merged,parent_avg)", "mean_abs_drift"],
            floatfmt=".5f",
            tablefmt="github",
        )
    )


@app.command("plots")
def cmd_plots(
    results_dir: Annotated[Path, typer.Option(help="results dir")] = Path("results"),
    figures_dir: Annotated[Path, typer.Option(help="figures dir")] = Path("results/figures"),
) -> None:
    plot_method_comparison(results_dir, figures_dir / "method_comparison.png")
    plot_layer_drift_heatmap(results_dir, figures_dir / "layer_drift_heatmap.png")
    plot_cosine_scatter(results_dir, figures_dir / "cosine_scatter.png")
    plot_top_layers_grouped(results_dir, figures_dir / "top_layers_grouped.png")
    plot_drift_bar(results_dir, figures_dir / "drift_bar.png")
    typer.echo(f"wrote 5 figures to {figures_dir}")


if __name__ == "__main__":
    app()
