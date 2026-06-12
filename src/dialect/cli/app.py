"""DIALECT's command-line interface: the Typer app + console-script entry point."""

from pathlib import Path

import typer

from dialect import api
from dialect.cli.simulate import simulate_app
from dialect.utils.logger import configure_logging

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="DIALECT: mutual exclusivity & co-occurrence between cancer drivers, "
    "accounting for the background mutation rate.",
)
app.add_typer(simulate_app, name="simulate")


@app.callback()
def _main(
    *,
    verbose: bool = typer.Option(
        False,
        "-v",
        "--verbose",
        help="Enable verbose logging.",
    ),
) -> None:
    """DIALECT command-line interface."""
    configure_logging(verbose=verbose)


@app.command()
def generate(
    maf: Path = typer.Option(..., "-m", "--maf", help="Input MAF file."),
    out: Path = typer.Option(..., "-o", "--out", help="Output directory."),
    reference: str = typer.Option(
        "hg19",
        "-r",
        "--reference",
        help="Reference genome (hg19 or hg38).",
    ),
    threshold: str = typer.Option(
        "1e-100",
        "-t",
        "--threshold",
        help="CBaSE BMR tail-truncation threshold.",
    ),
    bmr: str = typer.Option(
        "cbase",
        "--bmr",
        help="Background mutation rate provider (cbase or dig).",
    ),
    dig_results: str | None = typer.Option(
        None,
        "--dig-results",
        help="DIG geneDriver *.results.txt (required when --bmr dig).",
    ),
    dig_samples: int | None = typer.Option(
        None,
        "--dig-samples",
        help="Number of cohort samples (required when --bmr dig).",
    ),
) -> None:
    """Generate the background mutation rate and count matrix from a MAF."""
    if bmr == "dig":
        if not dig_results or not dig_samples:
            msg = "--bmr dig requires --dig-results and --dig-samples"
            raise typer.BadParameter(msg)
        api.estimate_bmr(
            maf,
            out,
            provider="dig",
            reference=reference,
            dig_results=dig_results,
            n_samples=dig_samples,
        )
    else:
        api.estimate_bmr(
            maf,
            out,
            provider="cbase",
            reference=reference,
            threshold=threshold,
        )


@app.command()
def identify(
    cnt: Path = typer.Option(..., "-c", "--cnt", help="Count matrix CSV."),
    bmr: Path = typer.Option(..., "-b", "--bmr", help="BMR PMFs CSV."),
    out: Path = typer.Option(..., "-o", "--out", help="Output directory."),
    top_k: int = typer.Option(
        100,
        "-k",
        "--top_k",
        help="Number of top genes (by count) to pair.",
    ),
    cbase_stats: Path | None = typer.Option(
        None,
        "-cb",
        "--cbase_stats",
        help="Optional CBaSE q_values.txt for positive-selection annotation.",
    ),
) -> None:
    """Identify mutual-exclusivity / co-occurrence interactions."""
    api.identify_interactions(
        counts=cnt,
        bmr_pmfs=bmr,
        out_dir=out,
        top_k=top_k,
        cbase_stats=cbase_stats,
    )


@app.command()
def compare(
    cnt: Path = typer.Option(..., "-c", "--cnt", help="Count matrix CSV."),
    out: Path = typer.Option(..., "-o", "--out", help="Output directory."),
    top_k: int = typer.Option(100, "-k", "--top_k", help="Number of top genes."),
    gene_level: bool = typer.Option(
        False,
        "-g",
        "--gene_level",
        help="Run comparison methods on gene-level features.",
    ),
) -> None:
    """Run the alternative ME/CO comparison methods (Fisher/DISCOVER/MEGSA/WeSME)."""
    api.compare_methods(cnt, out, top_k=top_k, gene_level=gene_level)


@app.command()
def merge(
    dialect: Path = typer.Option(..., "-d", "--dialect", help="DIALECT results CSV."),
    alt: Path = typer.Option(
        ...,
        "-a",
        "--alt",
        help="Alternative-method results CSV.",
    ),
    out: Path = typer.Option(..., "-o", "--out", help="Output directory."),
) -> None:
    """Merge DIALECT results with an alternative method's results."""
    api.merge_results(dialect, alt, out)


def main() -> None:
    """Entry point for the ``dialect`` console script."""
    app()


if __name__ == "__main__":
    main()
