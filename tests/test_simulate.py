"""Reproducibility tests for simulation generation."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from dialect.experiments import simulate

if TYPE_CHECKING:
    from pathlib import Path


def test_pair_driver_draw_uses_random_eligible_subset_and_exact_rng() -> None:
    """The eligible fraction is random, reproducible, and not a leading slice."""
    first = simulate.simulate_pairwise_gene_driver_mutations(
        0.0,
        1.0,
        0.0,
        20,
        0.4,
        rng=np.random.default_rng(17),
    )
    second = simulate.simulate_pairwise_gene_driver_mutations(
        0.0,
        1.0,
        0.0,
        20,
        0.4,
        rng=np.random.default_rng(17),
    )

    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert first[0].sum() == 8
    assert first[1].sum() == 0
    assert not np.array_equal(np.flatnonzero(first[0]), np.arange(8))


@pytest.mark.parametrize(
    ("taus", "driver_proportion", "match"),
    [
        ((0.8, 0.4, 0.0), 1.0, "summing to at most 1"),
        ((0.2, 0.2, 0.2), -0.1, "proportion"),
        ((0.2, 0.2, 0.2), 1.1, "proportion"),
    ],
)
def test_pair_driver_draw_rejects_invalid_probabilities(
    taus: tuple[float, float, float],
    driver_proportion: float,
    match: str,
) -> None:
    """Invalid latent weights or eligibility fractions fail before random draws."""
    with pytest.raises(ValueError, match=match):
        simulate.simulate_pairwise_gene_driver_mutations(
            *taus,
            10,
            driver_proportion,
            rng=np.random.default_rng(1),
        )


def test_single_simulation_seed_controls_output_bytes(tmp_path: Path) -> None:
    """Equal seeds reproduce exact NPY bytes; a different seed changes them."""
    outputs = [tmp_path / name for name in ("a", "b", "c")]
    for output, seed in zip(outputs, (11, 11, 12), strict=True):
        simulate.create_single_gene_simulation(
            pi=0.35,
            num_samples=100,
            num_simulations=4,
            length=100,
            mu=0.01,
            out=output,
            seed=seed,
        )

    filename = "single_gene_simulated_data.npy"
    assert (outputs[0] / filename).read_bytes() == (outputs[1] / filename).read_bytes()
    assert (outputs[0] / filename).read_bytes() != (outputs[2] / filename).read_bytes()
    params = json.loads(
        (outputs[0] / "single_gene_simulation_parameters.json").read_text(),
    )
    assert params["seed"] == 11
    assert params["rng_contract"] == simulate.RNG_CONTRACT


def test_pair_simulation_seed_controls_output_bytes(tmp_path: Path) -> None:
    """Every passenger and latent-driver draw shares the recorded generator."""
    outputs = [tmp_path / name for name in ("a", "b", "c")]
    for output, seed in zip(outputs, (21, 21, 22), strict=True):
        simulate.create_pair_gene_simulation(
            tau_10=0.2,
            tau_01=0.15,
            tau_11=0.1,
            num_samples=100,
            num_simulations=4,
            length_a=100,
            mu_a=0.01,
            length_b=120,
            mu_b=0.01,
            out=output,
            driver_proportion=0.65,
            seed=seed,
        )

    filename = "pair_gene_simulated_data.npy"
    assert (outputs[0] / filename).read_bytes() == (outputs[1] / filename).read_bytes()
    assert (outputs[0] / filename).read_bytes() != (outputs[2] / filename).read_bytes()
    params = json.loads(
        (outputs[0] / "pair_gene_simulation_parameters.json").read_text(),
    )
    assert params["seed"] == 21
    assert params["driver_proportion"] == 0.65
    assert params["rng_contract"] == simulate.RNG_CONTRACT


def _write_matrix_inputs(root: Path) -> tuple[Path, Path, Path]:
    count_path = root / "counts.csv"
    pmf_path = root / "pmfs.csv"
    driver_path = root / "drivers.tsv"
    counts = pd.DataFrame(
        {
            "A_M": [1, 0, 0, 0],
            "B_M": [0, 1, 0, 0],
            "C_M": [0, 0, 1, 0],
            "D_M": [0, 0, 0, 1],
            "P_M": [1, 1, 0, 0],
        },
        index=["s1", "s2", "s3", "s4"],
    )
    counts.to_csv(count_path)
    pd.DataFrame(
        {0: [0.9] * len(counts.columns), 1: [0.1] * len(counts.columns)},
        index=counts.columns,
    ).to_csv(pmf_path)
    pd.DataFrame({"label": [1, 1, 1, 1]}, index=["A", "B", "C", "D"]).to_csv(
        driver_path,
        sep="\t",
    )
    return count_path, pmf_path, driver_path


def test_matrix_simulation_records_seed_and_is_byte_reproducible(
    tmp_path: Path,
) -> None:
    """Feature selection, tau draws, and count draws all share one seeded stream."""
    count_path, pmf_path, driver_path = _write_matrix_inputs(tmp_path)
    outputs = [tmp_path / "first", tmp_path / "second"]
    for output in outputs:
        simulate.create_matrix_simulation(
            cnt_mtx_fn=count_path,
            bmr_pmfs_fn=pmf_path,
            driver_genes_fn=driver_path,
            dout=output,
            num_likely_passengers=1,
            num_me_pairs=1,
            num_co_pairs=1,
            num_samples=50,
            tau_uv_low=0.049,
            tau_uv_high=0.051,
            driver_proportion=0.65,
            seed=31,
        )

    assert (outputs[0] / "count_matrix.csv").read_bytes() == (
        outputs[1] / "count_matrix.csv"
    ).read_bytes()
    info = json.loads((outputs[0] / "matrix_simulation_info.json").read_text())
    assert info["seed"] == 31
    assert info["rng_contract"] == simulate.RNG_CONTRACT
