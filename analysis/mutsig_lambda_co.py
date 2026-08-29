"""DIALECT with the PROPER per-(gene,sample) MutSig BMR (patched-source lambda dump).

Reads ``persample_lambda.f32`` -- the per-(gene,patient,effect) expected background
that MutSig2CV computes *internally* (per-patient, per-category hypergeometric means),
dumped by our Octave-patched source (scripts/run_mutsig_octave.sh) -- and builds a
per-sample Poisson count PMF  P(B_{g,s}=k) = Poisson(lambda_{g,s,eff})  for DIALECT.

This replaces the scalar-f_p reconstruction in ``mutsig_persample_co.py``: lambda here
varies by sample AND sequence context (a POLE/MSI sample's excess flows through its
own per-category counts), which the single per-patient scalar could not represent.

The standalone command is deprecated for production unless the exact tail-bounded
support contract is supplied explicitly. Prefer ``run_tcga_revision_k500``. Usage::

    python analysis/mutsig_lambda_co.py --cohort UCEC \
        --results-root output --mutsig-root output/mutsigsrc -k 500 \
        --production-support-contract /path/to/cohort-contract.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy.stats import poisson

from analysis.bmr_fdr_comparison import call_significant, summarize
from dialect.models.assembly import initialize_interaction_objects
from dialect.models.gene import Gene
from dialect.utils.identify import (
    create_pairwise_results,
    estimate_pi_for_each_gene,
    estimate_taus_for_each_interaction,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

_FDR = 0.05
_EFF = {"M": 0, "N": 1}  # page index into the lambda dump
PRODUCTION_POISSON_TAIL_TOLERANCE = 1e-12
PRODUCTION_POISSON_SUPPORT_CONTRACT = (
    "native-float32-shared-cohort-inclusive-poisson-tail-v1"
)
PRODUCTION_POISSON_SUPPORT_RULE = (
    "K=max(observed_kmax,min{k>=0:Poisson.sf(k,max_selected_native_lambda)<=1e-12})"
)
PRODUCTION_POISSON_NORMALIZATION = "per-sample-math.fsum-v1"
PRODUCTION_POISSON_STORAGE_CONTRACT = (
    "platform-independent-numeric-array-estimate-with-row-mapping-views-v2"
)


class _PoissonPmfRow(Mapping[int, float]):
    """Read-only mapping view over one row of a shared dense PMF matrix."""

    __slots__ = ("_matrix", "_row")

    def __init__(self, matrix: np.ndarray, row: int) -> None:
        self._matrix = matrix
        self._row = row

    def __getitem__(self, key: int) -> float:
        if (
            not isinstance(key, (int, np.integer))
            or isinstance(key, bool)
            or key < 0
            or key >= self._matrix.shape[1]
        ):
            raise KeyError(key)
        return float(self._matrix[self._row, int(key)])

    def __iter__(self) -> Iterator[int]:
        return iter(range(self._matrix.shape[1]))

    def __len__(self) -> int:
        return self._matrix.shape[1]


def estimate_native_poisson_pmf_storage(
    feature_count: int,
    sample_count: int,
    inclusive_support_k: int,
) -> dict[str, Any]:
    """Estimate production numeric-array storage without platform object sizes."""
    coordinates = (feature_count, sample_count, inclusive_support_k)
    if (
        any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in coordinates
        )
        or feature_count == 0
        or sample_count == 0
    ):
        msg = "MutSig PMF storage coordinates must have positive axes and K >= 0."
        raise ValueError(msg)
    value_bytes = np.dtype(np.float64).itemsize
    dense_probability_bytes = (
        feature_count * sample_count * (inclusive_support_k + 1) * value_bytes
    )
    selected_native_rate_bytes = feature_count * sample_count * np.dtype("<f4").itemsize
    per_feature_matrix_bytes = sample_count * (inclusive_support_k + 1) * value_bytes
    support_vector_bytes = (inclusive_support_k + 1) * np.dtype(np.int64).itemsize
    normalizer_array_bytes = sample_count * value_bytes
    estimated_peak_numeric_array_bytes = (
        dense_probability_bytes
        + selected_native_rate_bytes
        + 2 * per_feature_matrix_bytes
        + support_vector_bytes
        + normalizer_array_bytes
    )
    return {
        "contract": PRODUCTION_POISSON_STORAGE_CONTRACT,
        "container_overhead": (
            "excluded-platform-dependent; completed task manifests record measured "
            "process peak RSS"
        ),
        "dense_probability_bytes": dense_probability_bytes,
        "estimated_peak_numeric_array_bytes": estimated_peak_numeric_array_bytes,
        "estimated_persistent_numeric_array_bytes": dense_probability_bytes,
        "feature_list_count": feature_count,
        "feature_count": feature_count,
        "inclusive_support_k": inclusive_support_k,
        "legacy_per_probability_dict_entries_materialized": False,
        "normalizer_array_bytes": normalizer_array_bytes,
        "per_feature_matrix_bytes": per_feature_matrix_bytes,
        "probability_dtype": "float64",
        "probability_value_count": (
            feature_count * sample_count * (inclusive_support_k + 1)
        ),
        "row_mapping_view_count": feature_count * sample_count,
        "sample_count": sample_count,
        "selected_native_rate_bytes": selected_native_rate_bytes,
        "scope": (
            "deterministic numeric arrays for selected native rates, PMF storage, "
            "and one-feature construction workspace"
        ),
        "support_vector_bytes": support_vector_bytes,
    }


def _smallest_poisson_tail_k(max_native_lambda: float, tolerance: float) -> int:
    """Return the smallest nonnegative K whose upper Poisson tail is bounded."""
    lower = -1
    upper = max(0, math.ceil(max_native_lambda))
    while float(poisson.sf(upper, max_native_lambda)) > tolerance:
        upper = max(1, 2 * upper)
    while upper - lower > 1:
        midpoint = (lower + upper) // 2
        if float(poisson.sf(midpoint, max_native_lambda)) <= tolerance:
            upper = midpoint
        else:
            lower = midpoint
    return upper


def build_poisson_support_contract(
    max_native_lambda: float,
    observed_kmax: int,
) -> dict[str, Any]:
    """Build the deterministic production support contract for native lambdas."""
    if (
        not isinstance(observed_kmax, int)
        or isinstance(observed_kmax, bool)
        or observed_kmax < 0
    ):
        msg = "Observed MutSig count maximum must be a nonnegative integer."
        raise ValueError(msg)
    max_error = (
        "Maximum native MutSig lambda must be finite, nonnegative, and "
        "exactly binary32-representable."
    )
    if isinstance(max_native_lambda, (bool, np.bool_)):
        raise TypeError(max_error)
    try:
        candidate_max = float(max_native_lambda)
    except TypeError:
        raise TypeError(max_error) from None
    except (OverflowError, ValueError):
        raise ValueError(max_error) from None
    if (
        not math.isfinite(candidate_max)
        or candidate_max < 0
        or candidate_max > float(np.finfo(np.float32).max)
    ):
        raise ValueError(max_error)
    native_max = np.float32(candidate_max)
    exact_max = float(native_max)
    if candidate_max != exact_max:
        raise ValueError(max_error)

    tail_tolerance = PRODUCTION_POISSON_TAIL_TOLERANCE
    tail_minimum_k = _smallest_poisson_tail_k(exact_max, tail_tolerance)
    inclusive_support_k = max(observed_kmax, tail_minimum_k)
    worst_tail = float(poisson.sf(inclusive_support_k, exact_max))
    predecessor_tail = float(poisson.sf(inclusive_support_k - 1, exact_max))
    tail_criterion_binds = inclusive_support_k == tail_minimum_k
    if worst_tail > tail_tolerance or (
        tail_criterion_binds and predecessor_tail <= tail_tolerance
    ):
        msg = "MutSig Poisson support is not the required minimal tail-bounded support."
        raise RuntimeError(msg)

    return {
        "contract": PRODUCTION_POISSON_SUPPORT_CONTRACT,
        "effect_pages": dict(_EFF),
        "feature_fallback": False,
        "inclusive_support_k": inclusive_support_k,
        "lambda_dtype": "float32",
        "lambda_floor": None,
        "max_selected_native_lambda": exact_max,
        "max_selected_native_lambda_float32_le_hex": np.asarray(
            [native_max],
            dtype="<f4",
        )
        .tobytes()
        .hex(),
        "native_lambda_only": True,
        "normalization": PRODUCTION_POISSON_NORMALIZATION,
        "observed_kmax": observed_kmax,
        "poisson_count_keys": [0, inclusive_support_k],
        "predecessor_tail_probability": predecessor_tail,
        "sample_fallback": False,
        "support_rule": PRODUCTION_POISSON_SUPPORT_RULE,
        "tail_criterion_binds": tail_criterion_binds,
        "tail_minimum_k": tail_minimum_k,
        "tail_tolerance": tail_tolerance,
        "tensor_dtype": "little-endian-float32",
        "tensor_order": "Fortran-(gene,patient,effect)",
        "worst_discarded_tail_probability": worst_tail,
    }


def validate_poisson_support_contract(
    contract: Mapping[str, object],
    *,
    max_native_lambda: float,
    observed_kmax: int,
) -> dict[str, Any]:
    """Return the canonical contract after an exact closed-schema comparison."""
    expected = build_poisson_support_contract(max_native_lambda, observed_kmax)
    if dict(contract) != expected:
        msg = "MutSig Poisson support contract drifted from the selected native rates."
        raise ValueError(msg)
    return expected


def build_native_poisson_pmfs(
    rates_by_feature: Mapping[str, np.ndarray],
    contract: Mapping[str, object],
) -> dict[str, list[Mapping[int, float]]]:
    """Build production PMFs with shared support and ``math.fsum`` normalization."""
    if not rates_by_feature:
        msg = "Production MutSig PMFs require at least one selected native feature."
        raise ValueError(msg)
    native_rates: dict[str, np.ndarray] = {}
    for feature, rates in rates_by_feature.items():
        values = np.asarray(rates)
        if values.dtype != np.dtype("<f4"):
            msg = (
                f"Production MutSig rates for {feature} are not the frozen "
                "little-endian float32 values."
            )
            raise TypeError(msg)
        if values.ndim != 1 or not np.isfinite(values).all() or (values < 0).any():
            msg = f"Production MutSig rates for {feature} are invalid."
            raise ValueError(msg)
        native_rates[feature] = values
    sample_counts = {len(values) for values in native_rates.values()}
    if len(sample_counts) != 1 or next(iter(sample_counts)) <= 0:
        msg = "Production MutSig feature rates are not equally sample-aligned."
        raise ValueError(msg)

    max_native_lambda = max(
        float(values.max(initial=np.float32(0))) for values in native_rates.values()
    )
    observed_kmax = contract.get("observed_kmax")
    if not isinstance(observed_kmax, int) or isinstance(observed_kmax, bool):
        msg = "MutSig Poisson support contract lacks an integer observed_kmax."
        raise TypeError(msg)
    canonical = validate_poisson_support_contract(
        contract,
        max_native_lambda=max_native_lambda,
        observed_kmax=observed_kmax,
    )
    support = np.arange(canonical["inclusive_support_k"] + 1, dtype=np.int64)
    pmfs: dict[str, list[Mapping[int, float]]] = {}
    for feature, rates in native_rates.items():
        raw = poisson.pmf(support[None, :], rates[:, None])
        normalizers = []
        for row in raw:
            total = math.fsum(float(value) for value in row)
            if not math.isfinite(total) or total <= 0:
                msg = f"Production MutSig PMF normalization failed for {feature}."
                raise ValueError(msg)
            normalizers.append(total)
        raw /= np.asarray(normalizers, dtype=np.float64)[:, None]
        raw.flags.writeable = False
        pmfs[feature] = [_PoissonPmfRow(raw, row) for row in range(len(rates))]
    return pmfs


def load_lambda(mutsig_dir: Path) -> tuple:
    """Read the raw per-(gene,patient,effect) lambda dump + its gene/patient labels."""
    if sys.byteorder != "little":
        msg = (
            "MutSig lambda is native-endian Octave fwrite output; standalone "
            "loading requires sys.byteorder == 'little'."
        )
        raise RuntimeError(msg)
    meta = dict(
        line.split("\t")
        for line in (mutsig_dir / "persample_meta.txt").read_text().splitlines()
        if line.strip()
    )
    ng, npat, neff = int(meta["ng"]), int(meta["np"]), int(meta["neff"])
    raw = (mutsig_dir / "persample_lambda.f32").read_bytes()
    expected_bytes = ng * npat * neff * np.dtype("<f4").itemsize
    if len(raw) != expected_bytes:
        msg = "MutSig lambda byte length does not match its metadata."
        raise ValueError(msg)
    lam = np.frombuffer(raw, dtype="<f4")
    lam = lam.reshape((ng, npat, neff), order="F")  # column-major from Octave fwrite
    lam.flags.writeable = False
    genes = (mutsig_dir / "persample_genes.txt").read_text().split()
    patients = (mutsig_dir / "persample_patients.txt").read_text().split()
    return lam, genes, patients


def build_lambda_pmfs(  # noqa: PLR0913
    gene_effects: list,
    samples: pd.Index,
    mutsig_dir: Path,
    cbase_pmfs: dict | None,
    kmax: int,
    *,
    allow_cbase_fallback: bool = True,
    require_all_features: bool = False,
    require_all_samples: bool = False,
    lambda_floor: float | None = 1e-12,
    production_contract: Mapping[str, object] | None = None,
) -> dict:
    """Build per-sample Poisson PMFs from native MutSig lambda values.

    The historical analysis allowed two conveniences: a feature absent from the
    MutSig gene axis could borrow its CBaSE PMF, and a sample absent from the MutSig
    patient axis could use the feature's cohort-mean lambda.  They remain the
    defaults for backward compatibility.  Frozen analyses can disable both and
    require complete native support with ``allow_cbase_fallback=False``,
    ``require_all_features=True``, and ``require_all_samples=True``. Set
    ``lambda_floor=None`` to preserve exact native zero rates; the historical
    default floor remains available only for backward compatibility. Passing an
    explicit ``production_contract`` additionally requires those strict settings,
    uses its cohort-shared tail-bounded support, and normalizes every sample PMF
    with ``math.fsum``.
    """
    if production_contract is not None and (
        allow_cbase_fallback
        or not require_all_features
        or not require_all_samples
        or lambda_floor is not None
    ):
        msg = (
            "Production MutSig PMFs require native features and samples with no "
            "lambda floor or fallback."
        )
        raise ValueError(msg)
    if production_contract is not None and (
        not isinstance(kmax, int)
        or isinstance(kmax, bool)
        or production_contract.get("observed_kmax") != kmax
    ):
        msg = "Production MutSig observed support drifted from its frozen contract."
        raise ValueError(msg)
    lam, genes, patients = load_lambda(mutsig_dir)
    if len(genes) != len(set(genes)):
        msg = "MutSig gene axis contains duplicate identifiers."
        raise ValueError(msg)
    if len(patients) != len(set(patients)):
        msg = "MutSig patient axis contains duplicate identifiers."
        raise ValueError(msg)
    gidx = {g: i for i, g in enumerate(genes)}
    pidx = {p: i for i, p in enumerate(patients)}
    col = [pidx.get(str(s), -1) for s in samples]  # mutsig patient column per sample
    missing_samples = [
        str(sample)
        for sample, position in zip(samples, col, strict=True)
        if position < 0
    ]
    if require_all_samples and missing_samples:
        msg = (
            "MutSig patient axis does not natively cover every count-matrix sample: "
            f"{missing_samples[:5]}"
        )
        raise ValueError(msg)
    historical_support = np.arange(kmax + 1) if production_contract is None else None

    out = {}
    production_rates: dict[str, np.ndarray] = {}
    missing_features = []
    for ge in gene_effects:
        base, eff = ge.rsplit("_", 1)
        gi = gidx.get(base)
        if gi is None or eff not in _EFF:
            if allow_cbase_fallback and cbase_pmfs is not None and ge in cbase_pmfs:
                out[ge] = dict(enumerate(cbase_pmfs[ge]))
            else:
                missing_features.append(ge)
            continue
        lam_g = lam[gi, :, _EFF[eff]]
        if production_contract is None:
            gene_mean = float(lam_g.mean()) if lam_g.size else 0.0
            lam_s = np.array(
                [lam_g[c] if c >= 0 else gene_mean for c in col],
                dtype=float,
            )
        else:
            lam_s = np.asarray(lam_g[np.asarray(col, dtype=np.intp)], dtype="<f4")
        if not np.isfinite(lam_s).all() or (lam_s < 0).any():
            msg = f"MutSig lambda contains an invalid native rate for {ge}."
            raise ValueError(msg)
        if lambda_floor is not None:
            if lambda_floor < 0:
                msg = "MutSig lambda floor must be nonnegative or None."
                raise ValueError(msg)
            lam_s = np.maximum(lam_s, lambda_floor)
        if production_contract is None:
            if historical_support is None:
                msg = "Historical MutSig support was not initialized."
                raise RuntimeError(msg)
            pmf = poisson.pmf(historical_support[None, :], lam_s[:, None])
            pmf /= pmf.sum(axis=1, keepdims=True)
            out[ge] = [dict(enumerate(row)) for row in pmf]
        else:
            production_rates[ge] = lam_s
    if require_all_features and missing_features:
        msg = (
            "MutSig lambda does not natively cover every requested feature: "
            f"{missing_features[:5]}"
        )
        raise ValueError(msg)
    if production_contract is not None:
        return build_native_poisson_pmfs(production_rates, production_contract)
    return out


def _build_genes(counts: pd.DataFrame, pmfs: dict, top_k: int) -> dict:
    totals = counts.sum(axis=0).sort_values(ascending=False)
    top = [g for g in totals.index if g in pmfs][:top_k]
    return {
        ge: Gene(
            name=ge,
            samples=counts.index,
            counts=counts[ge].to_numpy(),
            bmr_pmf=pmfs[ge],
        )
        for ge in top
    }


def run(  # noqa: PLR0913
    cohort: str,
    root: Path,
    mutsig_root: Path,
    top_k: int,
    suffix: str,
    *,
    production_contract: Mapping[str, object] | None = None,
) -> Path:
    """Run only with an explicit production tail-support contract.

    The formerly implicit path emitted observed-kmax-truncated PMFs. It is now a
    hard error so no production-facing invocation can silently reproduce them.
    """
    if production_contract is None:
        msg = (
            "Standalone MutSig execution is deprecated without an explicit "
            "production Poisson support contract; use run_tcga_revision_k500."
        )
        raise RuntimeError(msg)
    cohort_dir = root / cohort
    counts = pd.read_csv(cohort_dir / "count_matrix.csv", index_col=0)
    kmax = int(counts.to_numpy().max())
    pmfs = build_lambda_pmfs(
        list(counts.columns),
        counts.index,
        mutsig_root / cohort,
        None,
        kmax,
        allow_cbase_fallback=False,
        require_all_features=True,
        require_all_samples=True,
        lambda_floor=None,
        production_contract=production_contract,
    )
    genes = _build_genes(counts, pmfs, top_k)

    out = cohort_dir / f"id_{suffix}"
    out.mkdir(parents=True, exist_ok=True)
    estimate_pi_for_each_gene(genes.values())
    _, interactions = initialize_interaction_objects(top_k, genes.values())
    estimate_taus_for_each_interaction(interactions)
    create_pairwise_results(interactions, str(out / "pairwise_interaction_results.csv"))
    return out / "pairwise_interaction_results.csv"


def main() -> None:
    """Run DIALECT with the proper MutSig lambda BMR and compare to other BMRs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--results-root", default="output")
    parser.add_argument("--mutsig-root", default="output/mutsigsrc")
    parser.add_argument("-k", "--top-k", type=int, default=500)
    parser.add_argument("--suffix", default="mutsiglam")
    parser.add_argument(
        "--production-support-contract",
        type=Path,
        required=True,
        help=(
            "JSON containing the exact production Poisson support contract, or a "
            "cohort contract with a mutsig_pmf_contract field"
        ),
    )
    args = parser.parse_args()
    root = Path(args.results_root)

    try:
        support_document = json.loads(
            args.production_support_contract.read_text(encoding="utf-8"),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        parser.error(f"invalid production support contract: {error}")
    if not isinstance(support_document, dict):
        parser.error("production support contract must be a JSON object")
    support_contract = support_document.get(
        "mutsig_pmf_contract",
        support_document,
    )
    if not isinstance(support_contract, dict):
        parser.error("mutsig_pmf_contract must be a JSON object")

    pw = run(
        args.cohort,
        root,
        Path(args.mutsig_root),
        args.top_k,
        args.suffix,
        production_contract=support_contract,
    )
    called = call_significant(pd.read_csv(pw), _FDR)
    rows = [summarize(args.cohort, args.suffix, called)]
    for bmr in ("mutsig", "cbase", "dig"):
        fn = root / args.cohort / f"id_{bmr}" / "pairwise_interaction_results.csv"
        if fn.exists():
            other = call_significant(pd.read_csv(fn), _FDR)
            rows.append(summarize(args.cohort, bmr, other))
    pd.set_option("display.width", 200)
    print(f"\nProper MutSig lambda BMR vs others -- {args.cohort} (BH q<{_FDR})\n")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
