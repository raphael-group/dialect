"""DIALECT ``simulate`` sub-commands (create / evaluate x single / pair / matrix).

Thin Typer pass-throughs to :mod:`dialect.experiments.simulate`.
"""

from pathlib import Path

import typer

from dialect.experiments import simulate as sim

simulate_app = typer.Typer(
    no_args_is_help=True,
    help="Run simulations for evaluation and benchmarking.",
)
create_app = typer.Typer(no_args_is_help=True, help="Create simulation data.")
evaluate_app = typer.Typer(no_args_is_help=True, help="Evaluate simulation data.")
simulate_app.add_typer(create_app, name="create")
simulate_app.add_typer(evaluate_app, name="evaluate")


@create_app.command("single")
def create_single(
    pi: float = typer.Option(..., "-pi", "--pi", help="Pi value (between 0 and 1)"),
    out: Path = typer.Option(..., "-o", "--out"),
    num_samples: int = typer.Option(1000, "-n", "--num_samples"),
    num_simulations: int = typer.Option(2500, "-ns", "--num_simulations"),
    length: int = typer.Option(10000, "-l", "--length"),
    mu: float = typer.Option(1e-6, "-m", "--mu"),
    seed: int = typer.Option(42, "-s", "--seed"),
) -> None:
    """Create single-gene simulations."""
    if not 0 <= pi <= 1:
        msg = "Value for --pi must be between 0 and 1"
        raise typer.BadParameter(msg)
    sim.create_single_gene_simulation(
        pi=pi,
        num_samples=num_samples,
        num_simulations=num_simulations,
        length=length,
        mu=mu,
        out=out,
        seed=seed,
    )


@create_app.command("pair")
def create_pair(
    tau_10: float = typer.Option(..., "-t10", "--tau_10"),
    tau_01: float = typer.Option(..., "-t01", "--tau_01"),
    tau_11: float = typer.Option(..., "-t11", "--tau_11"),
    out: Path = typer.Option(..., "-o", "--out"),
    num_samples: int = typer.Option(1000, "-n", "--num_samples"),
    num_simulations: int = typer.Option(2500, "-ns", "--num_simulations"),
    length_a: int = typer.Option(10000, "-la", "--length_a"),
    length_b: int = typer.Option(10000, "-lb", "--length_b"),
    mu_a: float = typer.Option(1e-6, "-ma", "--mu_a"),
    mu_b: float = typer.Option(1e-6, "-mb", "--mu_b"),
    seed: int = typer.Option(42, "-s", "--seed"),
) -> None:
    """Create pairwise-gene simulations."""
    sim.create_pair_gene_simulation(
        tau_10=tau_10,
        tau_01=tau_01,
        tau_11=tau_11,
        num_samples=num_samples,
        num_simulations=num_simulations,
        length_a=length_a,
        mu_a=mu_a,
        length_b=length_b,
        mu_b=mu_b,
        out=out,
        seed=seed,
    )


@create_app.command("matrix")
def create_matrix(
    cnt_mtx: Path = typer.Option(..., "-c", "--cnt_mtx"),
    bmr_pmfs: Path = typer.Option(..., "-b", "--bmr_pmfs"),
    driver_genes: Path = typer.Option(..., "-d", "--driver_genes"),
    out: Path = typer.Option(..., "-o", "--out"),
    num_me_pairs: int = typer.Option(..., "-nme", "--num_me_pairs"),
    num_co_pairs: int = typer.Option(..., "-nco", "--num_co_pairs"),
    tau_low: float = typer.Option(..., "-tl", "--tau_low"),
    tau_high: float = typer.Option(..., "-th", "--tau_high"),
    num_likely_passengers: int = typer.Option(100, "-nlp", "--num_likely_passengers"),
    driver_proportion: float = typer.Option(1.0, "-dp", "--driver_proportion"),
    num_samples: int = typer.Option(1000, "-n", "--num_samples"),
    seed: int = typer.Option(42, "-s", "--seed"),
) -> None:
    """Create matrix simulations."""
    sim.create_matrix_simulation(
        cnt_mtx_fn=cnt_mtx,
        bmr_pmfs_fn=bmr_pmfs,
        driver_genes_fn=driver_genes,
        dout=out,
        num_likely_passengers=num_likely_passengers,
        num_me_pairs=num_me_pairs,
        num_co_pairs=num_co_pairs,
        num_samples=num_samples,
        tau_uv_low=tau_low,
        tau_uv_high=tau_high,
        driver_proportion=driver_proportion,
        seed=seed,
    )


@evaluate_app.command("single")
def evaluate_single(
    params: Path = typer.Option(..., "-p", "--params"),
    data: Path = typer.Option(..., "-d", "--data"),
    out: str = typer.Option(..., "-o", "--out"),
) -> None:
    """Evaluate single-gene simulations."""
    sim.evaluate_single_gene_simulation(params=params, data=data, out=out)


@evaluate_app.command("pair")
def evaluate_pair(
    params: Path = typer.Option(..., "-p", "--params"),
    data: Path = typer.Option(..., "-d", "--data"),
    out: str = typer.Option(..., "-o", "--out"),
) -> None:
    """Evaluate pairwise-gene simulations."""
    sim.evaluate_pair_gene_simulation(params=params, data=data, out=out)


@evaluate_app.command("matrix")
def evaluate_matrix(
    results: Path = typer.Option(..., "-r", "--results"),
    num_runs: int = typer.Option(..., "-n", "--num_runs"),
    ixn_type: str = typer.Option(..., "-ixn", "--ixn_type"),
    out: Path = typer.Option(..., "-o", "--out"),
) -> None:
    """Evaluate matrix simulations."""
    sim.evaluate_matrix_simulation(
        results_dir=results,
        out=out,
        nruns=num_runs,
        ixn_type=ixn_type,
    )
