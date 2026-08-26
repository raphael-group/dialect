"""Build the complete, immutable K=100 DIALECT Atlas data release.

The release is deliberately separate from the historical paper tables. It contains
every evaluated DIALECT pair for each cohort and BMR, a freshly generated comparison
table for the same cohort's top event features, and provenance for every file.

Statistical contract
--------------------
* ``p = chi2.sf(max(LRT, 0), 1)``.
* Benjamini-Hochberg is applied once across all evaluated DIALECT pairs within a
  cohort/BMR (one family shared by mutually exclusive and co-occurring pairs).
* ``q < 0.01`` is significant; direction is the sign of Marshall-Olkin rho.
* No epsilon filter is applied. Same-base-gene pairs remain in the test family and
  are excluded only by the presentation layer.
* ME ranks by rho ascending; CO ranks by raw LRT descending.

Usage::

    python -m analysis.build_atlas_data \
      --out atlas/public/data/releases/k100-2026-08-26 \
      --baseline-root output/atlas_baselines/k100 \
      --release-id k100-2026-08-26 \
      --generated-at 2026-08-26T00:00:00Z
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import tempfile
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests

RELEASE_ID = "k100-2026-08-26"
SCHEMA_VERSION = "2.0.0"
TOP_K = 100
FDR = 0.01
DISCOVER_VERSION = "0.9.6"
DISCOVER_COMMIT = "a46d99f9a8a76dc6302f42c814650ca2a1568267"
DISCOVER_SOURCE_SHA256 = (
    "117ea3646653e7fafd1311e94f5fc62c8500ea3eb2f22eabb4fa8d5b109d5e3c"
)

BMRS = ("cbase", "dig", "mutsig")
BMR_METADATA = {
    "cbase": {"label": "CBaSE", "role": "primary"},
    "dig": {"label": "DIG", "role": "robustness"},
    "mutsig": {"label": "MutSigCV2", "role": "robustness"},
}

MUTSIG_INPUT_NAMES = (
    "persample_lambda.f32",
    "persample_meta.txt",
    "persample_genes.txt",
    "persample_patients.txt",
)
PUBLISHED_FEATURE_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*_[MN]")
RAW_FEATURE_ID_PATTERN = re.compile(r".+_[MN]")
GENE_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
NONNEGATIVE_INTEGER_PATTERN = re.compile(r"[0-9]+")
PMF_SUM_TOLERANCE = 1e-6
MUTSIG_EFFECT_COUNT = 2
FLOAT32_BYTES = 4
VALIDATION_CHUNK_VALUES = 4 * 1024 * 1024
INFERENCE_SOURCE_FILES = (
    Path("src/dialect/models/assembly.py"),
    Path("src/dialect/models/gene.py"),
    Path("src/dialect/models/interaction.py"),
    Path("src/dialect/utils/identify.py"),
    Path("analysis/mutsig_lambda_co.py"),
)
RELEASE_SOURCE_FILES = (
    Path("analysis/build_atlas_data.py"),
    Path("analysis/build_atlas_baselines.py"),
    *INFERENCE_SOURCE_FILES,
)

DIALECT_SOURCE_COLUMNS = (
    "Gene A",
    "Gene B",
    "Tau_00",
    "Tau_10",
    "Tau_01",
    "Tau_11",
    "_00_",
    "_10_",
    "_01_",
    "_11_",
    "Tau_1X",
    "Tau_X1",
    "Rho",
    "Log Odds Ratio",
    "Likelihood Ratio",
    "Wald Statistic",
)
DIALECT_FIELDS = (
    "ga",
    "gb",
    "tau00",
    "tau10",
    "tau01",
    "tau11",
    "observed_both",
    "observed_b_only",
    "observed_a_only",
    "observed_neither",
    "tau1x",
    "taux1",
    "rho",
    "log_odds_ratio",
    "lrt",
    "wald",
    "p",
    "q",
    "direction",
    "rank",
    "tau_mass",
    "effective_n",
    "excluded_samples",
)

BASELINE_COLUMN_MAP = {
    "Gene A": "ga",
    "Gene B": "gb",
    "Fisher's ME P-Val": "fisher_me_p",
    "Fisher's CO P-Val": "fisher_co_p",
    "Fisher's ME Q-Val": "fisher_me_q",
    "Fisher's CO Q-Val": "fisher_co_q",
    "Discover ME P-Val": "discover_me_p",
    "Discover CO P-Val": "discover_co_p",
    "Discover ME Q-Val": "discover_me_q",
    "Discover CO Q-Val": "discover_co_q",
    "MEGSA S-Score (LRT)": "megsa_lrt",
    "MEGSA P-Val": "megsa_p",
    "MEGSA Q-Val": "megsa_q",
    "WeSME P-Val": "wesme_p",
    "WeSCO P-Val": "wesco_p",
    "WeSME Q-Val": "wesme_q",
    "WeSCO Q-Val": "wesco_q",
}
BASELINE_FIELDS = tuple(BASELINE_COLUMN_MAP.values())
BASELINE_PROBABILITY_FIELDS = tuple(
    field for field in BASELINE_FIELDS if field.endswith(("_p", "_q"))
)

TCGA_CANCER_NAMES = {
    "ACC": "Adrenocortical carcinoma",
    "BLCA": "Bladder urothelial carcinoma",
    "BRCA": "Breast invasive carcinoma",
    "CESC": "Cervical squamous cell carcinoma",
    "CHOL": "Cholangiocarcinoma",
    "CRAD": "Colorectal adenocarcinoma",
    "DLBC": "Diffuse large B-cell lymphoma",
    "ESCA": "Esophageal carcinoma",
    "GBM": "Glioblastoma",
    "HNSC": "Head and neck squamous cell carcinoma",
    "KICH": "Kidney chromophobe",
    "KIRC": "Kidney clear cell carcinoma",
    "KIRP": "Kidney papillary cell carcinoma",
    "LAML": "Acute myeloid leukemia",
    "LGG": "Lower-grade glioma",
    "LIHC": "Liver hepatocellular carcinoma",
    "LUAD": "Lung adenocarcinoma",
    "LUSC": "Lung squamous cell carcinoma",
    "MESO": "Mesothelioma",
    "OV": "Ovarian serous cystadenocarcinoma",
    "PAAD": "Pancreatic adenocarcinoma",
    "PCPG": "Pheochromocytoma and paraganglioma",
    "PRAD": "Prostate adenocarcinoma",
    "SARC": "Sarcoma",
    "SKCM": "Skin cutaneous melanoma",
    "STAD": "Stomach adenocarcinoma",
    "TGCT": "Testicular germ cell tumor",
    "THCA": "Thyroid carcinoma",
    "THYM": "Thymoma",
    "UCEC": "Uterine endometrial carcinoma",
    "UCS": "Uterine carcinosarcoma",
    "UVM": "Uveal melanoma",
}

EXPECTED_COHORTS = {
    "TCGA": tuple(TCGA_CANCER_NAMES),
    "MSK-IMPACT": (
        "Ampullary_Cancer",
        "Anal_Cancer",
        "Appendiceal_Cancer",
        "Bladder_Cancer",
        "Bone_Cancer",
        "Breast_Cancer",
        "CNS_Cancer",
        "Cancer_of_Unknown_Primary",
        "Cervical_Cancer",
        "Colorectal_Cancer",
        "Endometrial_Cancer",
        "Esophagogastric_Cancer",
        "Gastrointestinal_Neuroendocrine_Tumor",
        "Gastrointestinal_Stromal_Tumor",
        "Germ_Cell_Tumor",
        "Glioma",
        "Head_and_Neck_Cancer",
        "Hepatobiliary_Cancer",
        "Melanoma",
        "Mesothelioma",
        "Nerve_Sheath_Tumor",
        "Non_Small_Cell_Lung_Cancer",
        "Ovarian_Cancer",
        "Pancreatic_Cancer",
        "Peripheral_Nervous_System",
        "Prostate_Cancer",
        "Renal_Cell_Carcinoma",
        "Salivary_Gland_Cancer",
        "Skin_Cancer_Non_Melanoma",
        "Small_Bowel_Cancer",
        "Small_Cell_Lung_Cancer",
        "Soft_Tissue_Sarcoma",
        "Thyroid_Cancer",
        "Uterine_Sarcoma",
    ),
    "MSK-CHORD": (
        "Breast_Cancer",
        "Colorectal_Cancer",
        "Non_Small_Cell_Lung_Cancer",
        "Pancreatic_Cancer",
        "Prostate_Cancer",
    ),
}
EXPECTED_COHORT_IDS = tuple(
    sorted(
        f"{study}__{cohort}"
        for study, cohort_names in EXPECTED_COHORTS.items()
        for cohort in cohort_names
    ),
)


@dataclass(frozen=True)
class Source:
    """One study collection containing cohort output directories."""

    study: str
    root: Path
    mutsig_root: Path | None = None


DEFAULT_SOURCES = (
    Source("TCGA", Path("output/pancan"), Path("output/mutsigsrc")),
    Source(
        "MSK-IMPACT",
        Path("output/msk/IMPACT2026"),
        Path("output/mutsigsrc_msk/IMPACT2026"),
    ),
    Source(
        "MSK-CHORD",
        Path("output/msk/CHORD2024"),
        Path("output/mutsigsrc_msk/CHORD2024"),
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _require_committed_release_sources(paths: tuple[Path, ...]) -> None:
    """Fail unless every release source is tracked and byte-identical to HEAD."""
    for path in paths:
        try:
            committed = subprocess.run(
                ["git", "show", f"HEAD:{path.as_posix()}"],
                check=True,
                capture_output=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError) as error:
            msg = f"release source is not committed at HEAD: {path}"
            raise RuntimeError(msg) from error
        if not path.is_file() or path.read_bytes() != committed:
            msg = f"release source differs from HEAD: {path}"
            raise RuntimeError(msg)


def _sequence_sha256(values: list[str] | tuple[str, ...]) -> str:
    payload = json.dumps(
        list(values),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _finite_or_none(value: object) -> object:
    """Convert pandas/numpy scalars to strict-JSON-safe Python values."""
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value.item() if isinstance(value, np.generic) else value


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
        encoding="utf-8",
    )


def _write_release_readme(path: Path, release_id: str) -> None:
    """Write the human-readable data dictionary shipped with the release."""
    path.write_text(
        f"""# DIALECT Atlas {release_id}

Complete K=100 release for 71 TCGA, MSK-IMPACT, and MSK-CHORD cohorts.

## Files

- `manifest.json`: immutable analysis contract, coverage, methods, and provenance.
- `index.json`: cohort metadata, summaries, paths, byte counts, and SHA-256 hashes.
- `cohorts/<study>__<cohort>.json`: every evaluated DIALECT pair for CBaSE,
  DIG, and MutSigCV2 plus every Fisher, DISCOVER, MEGSA, and WeSME/WeSCO result.

Compact tables store a `fields` array followed by positional `rows`. Gene-effect
suffixes are `_M` (missense) and `_N` (nonsense). Observed contingency fields are
named semantically: `observed_both`, `observed_b_only`, `observed_a_only`, and
`observed_neither`.

## DIALECT inference

For every cohort and BMR, `p = chi2.sf(max(LRT, 0), df=1)` and Benjamini-Hochberg
is applied once across all evaluated pairs. ME and CO share the same correction
family. `q < 0.01` is significant. Direction comes from rho (`rho < 0`: ME;
`rho > 0`: CO). No epsilon filter is applied. ME ranks by rho ascending; CO ranks
by raw LRT descending. Raw negative numerical LRT values are preserved but carry
zero evidence for the p-value.

Tau values are preserved exactly as fitted. When a pair has samples with no
background-PMF support, those samples contribute zero posterior mass in the
historical EM implementation. `tau_mass` then equals `effective_n / cohort_n`
rather than one; `effective_n` and `excluded_samples` expose that condition for
every pair. The release does not renormalize these values, and such fits can be
biased, particularly for hypermutated samples.

The default web view requires the exact gene-effect pair and direction to agree
across all three BMRs. FDR support is reported separately; strict consensus means
`q < 0.01` under all three.

Each BMR can cover a different set of gene-effects because a provider may not emit
a background PMF for every feature. K=100 therefore means the top 100 count-ranked
features available to that BMR. Consensus is evaluated only where the exact pair
was tested by all three BMRs. A missing comparison-method value means that pair was
outside that method's separately selected count-ranked K=100 universe, not that
the method found negative evidence.

Equal-count ties preserve each historical producer's ordering: count-matrix column
order for CBaSE and DIG, and pandas descending quicksort order for MutSigCV2. Every
cohort publishes the selected ordered feature list and its SHA-256 digest.

The historical MutSig producer is hybrid: it uses per-sample MutSig lambda values
when a feature's base gene is present on the MutSig gene axis and otherwise falls
back to that cohort's CBaSE PMF. The release publishes both feature-origin lists.
Pairs touching a fallback feature remain available in the individual MutSig view
but are excluded from the default all-three consensus because they do not provide
a distinct-background third-model sensitivity check.

## Baseline calls

Fisher, DISCOVER, WeSME, and WeSCO use their emitted direction-specific BH
`q < 0.01`. MEGSA is ME-only and uses `p < 0.001`. Baselines are recomputed from
the existing count matrices and do not require a DIALECT or BMR rerun.

See `manifest.json` for exact source hashes, software versions, RNG seeds, and
method provenance. The historical DIALECT tables were produced across more than
one development snapshot and the original run commit was not recorded. The release
therefore does not attribute them to the current repository HEAD: it records exact
result/input hashes and timestamps plus the implementation snapshot used for release
assembly. The comparison-method tables were regenerated deterministically for this
release.
""",
        encoding="utf-8",
    )


def _base_gene(gene_effect: str) -> str:
    return gene_effect.rsplit("_", 1)[0]


def _pair_key(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((a, b)))


def _validate_complete_pair_universe(df: pd.DataFrame, *, label: str) -> list[str]:
    if df[["Gene A", "Gene B"]].isna().any().any():
        msg = f"{label}: null pair identifier"
        raise ValueError(msg)
    keys = [
        _pair_key(str(a), str(b))
        for a, b in zip(df["Gene A"], df["Gene B"], strict=True)
    ]
    if any(a == b for a, b in keys):
        msg = f"{label}: self-pair found"
        raise ValueError(msg)
    if len(keys) != len(set(keys)):
        msg = f"{label}: duplicate unordered pair"
        raise ValueError(msg)
    genes = sorted({gene for key in keys for gene in key})
    invalid_genes = [
        gene
        for gene in genes
        if PUBLISHED_FEATURE_ID_PATTERN.fullmatch(gene) is None
    ]
    if invalid_genes:
        msg = f"{label}: unsafe published gene-effect IDs: {invalid_genes[:5]}"
        raise ValueError(msg)
    expected_count = len(genes) * (len(genes) - 1) // 2
    if len(keys) != expected_count or set(keys) != set(combinations(genes, 2)):
        msg = (
            f"{label}: incomplete pair universe "
            f"({len(keys)} rows for {len(genes)} features)"
        )
        raise ValueError(msg)
    if len(genes) > TOP_K:
        msg = f"{label}: {len(genes)} features exceeds K={TOP_K}"
        raise ValueError(msg)
    return genes


def _rank_directions(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    rho = df["Rho"].astype(float)
    direction = np.full(len(df), "neutral", dtype=object)
    direction[rho < 0] = "ME"
    direction[rho > 0] = "CO"
    ranks = np.zeros(len(df), dtype=int)
    frame = df.assign(_position=np.arange(len(df)))
    me = frame[rho < 0].sort_values(
        ["Rho", "Gene A", "Gene B"],
        ascending=[True, True, True],
        kind="stable",
    )
    co = frame[rho > 0].sort_values(
        ["Likelihood Ratio", "Gene A", "Gene B"],
        ascending=[False, True, True],
        kind="stable",
    )
    ranks[me["_position"].to_numpy(dtype=int)] = np.arange(1, len(me) + 1)
    ranks[co["_position"].to_numpy(dtype=int)] = np.arange(1, len(co) + 1)
    return direction, ranks


def dialect_payload(
    path: Path,
    *,
    n_samples: int,
    label: str,
    counts: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Validate and encode one complete DIALECT pairwise result table."""
    df = pd.read_csv(path)
    missing = set(DIALECT_SOURCE_COLUMNS) - set(df.columns)
    if missing:
        msg = f"{label}: missing DIALECT columns: {sorted(missing)}"
        raise ValueError(msg)
    genes = _validate_complete_pair_universe(df, label=label)

    numeric_required = [*DIALECT_SOURCE_COLUMNS[2:13], "Likelihood Ratio"]
    if df[numeric_required].isna().any().any():
        msg = f"{label}: null required DIALECT statistic"
        raise ValueError(msg)
    if not np.isfinite(df[numeric_required].to_numpy(dtype=float)).all():
        msg = f"{label}: non-finite required DIALECT statistic"
        raise ValueError(msg)

    taus = df[["Tau_00", "Tau_10", "Tau_01", "Tau_11"]].to_numpy(dtype=float)
    tau_mass = taus.sum(axis=1)
    excluded_samples = np.rint(n_samples * (1 - tau_mass)).astype(int)
    effective_n = n_samples - excluded_samples
    expected_tau_mass = effective_n / n_samples
    if (
        ((taus < -1e-10) | (taus > 1 + 1e-10)).any()
        or (excluded_samples < 0).any()
        or (effective_n <= 0).any()
        or not np.allclose(tau_mass, expected_tau_mass, atol=1e-10, rtol=1e-10)
    ):
        msg = f"{label}: invalid latent tau probabilities"
        raise ValueError(msg)
    rho = df["Rho"].to_numpy(dtype=float)
    if ((rho < -1 - 1e-10) | (rho > 1 + 1e-10)).any():
        msg = f"{label}: rho outside [-1, 1]"
        raise ValueError(msg)
    tau00, tau10, tau01, tau11 = taus.T
    expected_tau1x = tau10 + tau11
    expected_taux1 = tau01 + tau11
    if not np.allclose(df["Tau_1X"], expected_tau1x, atol=1e-10, rtol=1e-10):
        msg = f"{label}: Tau_1X does not match Tau_10 + Tau_11"
        raise ValueError(msg)
    if not np.allclose(df["Tau_X1"], expected_taux1, atol=1e-10, rtol=1e-10):
        msg = f"{label}: Tau_X1 does not match Tau_01 + Tau_11"
        raise ValueError(msg)
    denominator = np.sqrt(
        (tau00 + tau01) * (tau10 + tau11) * (tau00 + tau10) * (tau01 + tau11),
    )
    if (denominator <= 0).any():
        msg = f"{label}: degenerate tau marginals cannot define rho"
        raise ValueError(msg)
    expected_rho = (tau11 * tau00 - tau01 * tau10) / denominator
    if not np.allclose(rho, expected_rho, atol=1e-10, rtol=1e-10):
        msg = f"{label}: rho does not match latent tau probabilities"
        raise ValueError(msg)
    log_odds = pd.to_numeric(df["Log Odds Ratio"], errors="coerce").to_numpy()
    wald = pd.to_numeric(df["Wald Statistic"], errors="coerce").to_numpy()
    defined = (tau01 * tau10 > 0) & (tau00 * tau11 > 0)
    expected_log_odds = np.full(len(df), np.nan)
    expected_wald = np.full(len(df), np.nan)
    expected_log_odds[defined] = np.log(
        (tau01[defined] * tau10[defined]) / (tau00[defined] * tau11[defined]),
    )
    expected_wald[defined] = expected_log_odds[defined] / np.sqrt(
        (1 / tau01[defined])
        + (1 / tau10[defined])
        + (1 / tau00[defined])
        + (1 / tau11[defined]),
    )
    if (np.isfinite(log_odds) != defined).any() or not np.allclose(
        log_odds[defined],
        expected_log_odds[defined],
        atol=1e-10,
        rtol=1e-10,
    ):
        msg = f"{label}: log odds ratio does not match latent tau probabilities"
        raise ValueError(msg)
    if (np.isfinite(wald) != defined).any() or not np.allclose(
        wald[defined],
        expected_wald[defined],
        atol=1e-10,
        rtol=1e-10,
    ):
        msg = f"{label}: Wald statistic does not match latent tau probabilities"
        raise ValueError(msg)

    count_columns = ["_00_", "_10_", "_01_", "_11_"]
    contingencies = df[count_columns].to_numpy(dtype=float)
    if (contingencies < 0).any() or not np.equal(
        contingencies,
        np.floor(contingencies),
    ).all():
        msg = f"{label}: invalid contingency count"
        raise ValueError(msg)
    if not np.equal(contingencies.sum(axis=1), n_samples).all():
        msg = f"{label}: contingency counts do not sum to n={n_samples}"
        raise ValueError(msg)
    if counts is not None:
        missing_count_features = set(genes) - set(counts.columns)
        if missing_count_features:
            msg = f"{label}: pair gene missing from count matrix"
            raise ValueError(msg)
        feature_position = {gene: position for position, gene in enumerate(genes)}
        binary = (counts[genes].to_numpy() > 0).astype(np.int64)
        cooccurrences = binary.T @ binary
        marginals = binary.sum(axis=0)
        a_position = df["Gene A"].map(feature_position).to_numpy(dtype=int)
        b_position = df["Gene B"].map(feature_position).to_numpy(dtype=int)
        observed_both = cooccurrences[a_position, b_position]
        observed_a_only = marginals[a_position] - observed_both
        observed_b_only = marginals[b_position] - observed_both
        observed_neither = (
            n_samples - marginals[a_position] - marginals[b_position] + observed_both
        )
        expected_contingencies = np.column_stack(
            (
                observed_both,
                observed_b_only,
                observed_a_only,
                observed_neither,
            ),
        )
        if not np.array_equal(contingencies, expected_contingencies):
            msg = f"{label}: contingency counts disagree with count matrix"
            raise ValueError(msg)

    raw_lrt = df["Likelihood Ratio"].to_numpy(dtype=float)
    p_values = chi2.sf(np.maximum(raw_lrt, 0), df=1)
    q_values = multipletests(p_values, method="fdr_bh")[1]
    direction, ranks = _rank_directions(df)

    rows: list[list[object]] = []
    for position, row in df.iterrows():
        rows.append(
            [
                str(row["Gene A"]),
                str(row["Gene B"]),
                _finite_or_none(row["Tau_00"]),
                _finite_or_none(row["Tau_10"]),
                _finite_or_none(row["Tau_01"]),
                _finite_or_none(row["Tau_11"]),
                int(row["_00_"]),
                int(row["_10_"]),
                int(row["_01_"]),
                int(row["_11_"]),
                _finite_or_none(row["Tau_1X"]),
                _finite_or_none(row["Tau_X1"]),
                _finite_or_none(row["Rho"]),
                _finite_or_none(row["Log Odds Ratio"]),
                _finite_or_none(row["Likelihood Ratio"]),
                _finite_or_none(row["Wald Statistic"]),
                float(p_values[position]),
                float(q_values[position]),
                str(direction[position]),
                int(ranks[position]),
                float(tau_mass[position]),
                int(effective_n[position]),
                int(excluded_samples[position]),
            ],
        )

    counts_by_direction = Counter(direction.tolist())
    significant = q_values < FDR
    summary = {
        "features": len(genes),
        "tested_pairs": len(df),
        "directions": {
            "ME": int(counts_by_direction["ME"]),
            "CO": int(counts_by_direction["CO"]),
            "neutral": int(counts_by_direction["neutral"]),
        },
        "significant_q_lt_0_01": {
            "ME": int(np.sum(significant & (direction == "ME"))),
            "CO": int(np.sum(significant & (direction == "CO"))),
            "neutral": int(np.sum(significant & (direction == "neutral"))),
        },
        "negative_lrt_count": int(np.sum(raw_lrt < 0)),
        "em_support": {
            "rows_with_excluded_samples": int(np.sum(excluded_samples > 0)),
            "max_excluded_samples": int(excluded_samples.max(initial=0)),
        },
    }
    return {
        "fields": list(DIALECT_FIELDS),
        "rows": rows,
        "summary": summary,
        "source": _hashed_file(path),
    }


def baseline_payload(path: Path, *, label: str) -> dict[str, object]:
    """Validate and encode one complete four-family comparison table."""
    df = pd.read_csv(path)
    missing = set(BASELINE_COLUMN_MAP) - set(df.columns)
    if missing:
        msg = f"{label}: incomplete baseline table; missing {sorted(missing)}"
        raise ValueError(msg)
    df = df[list(BASELINE_COLUMN_MAP)]
    genes = _validate_complete_pair_universe(df, label=label)
    if df.isna().any().any():
        missing_counts = df.isna().sum()
        detail = {key: int(value) for key, value in missing_counts.items() if value}
        msg = f"{label}: null baseline values: {detail}"
        raise ValueError(msg)
    numeric = df.iloc[:, 2:].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        msg = f"{label}: non-finite baseline statistic"
        raise ValueError(msg)
    for source, target in BASELINE_COLUMN_MAP.items():
        if target not in BASELINE_PROBABILITY_FIELDS:
            continue
        values = df[source].to_numpy(dtype=float)
        if ((values < 0) | (values > 1)).any():
            msg = f"{label}: {source} outside [0, 1]"
            raise ValueError(msg)

    rows = [
        [_finite_or_none(value) for value in row]
        for row in df.itertuples(index=False, name=None)
    ]
    summary = {
        "features": len(genes),
        "tested_pairs": len(df),
        "significant_calls": {
            "fisher_me": int((df["Fisher's ME Q-Val"] < FDR).sum()),
            "fisher_co": int((df["Fisher's CO Q-Val"] < FDR).sum()),
            "discover_me": int((df["Discover ME Q-Val"] < FDR).sum()),
            "discover_co": int((df["Discover CO Q-Val"] < FDR).sum()),
            "megsa_me": int((df["MEGSA P-Val"] < 0.001).sum()),
            "wesme_me": int((df["WeSME Q-Val"] < FDR).sum()),
            "wesco_co": int((df["WeSCO Q-Val"] < FDR).sum()),
        },
    }
    metadata_path = path.with_name("metadata.json")
    if not metadata_path.exists():
        msg = f"{label}: missing baseline metadata: {metadata_path}"
        raise FileNotFoundError(msg)
    metadata = json.loads(metadata_path.read_text())
    comparison_hash = _sha256(path)
    recorded_hash = metadata.get("artifacts", {}).get("comparison", {}).get("sha256")
    if recorded_hash != comparison_hash:
        msg = f"{label}: baseline comparison hash does not match its metadata"
        raise ValueError(msg)
    return {
        "fields": list(BASELINE_FIELDS),
        "rows": rows,
        "summary": summary,
        "source": {
            **_hashed_file(path),
            "metadata": metadata,
        },
    }


def _cancer_name(study: str, cohort: str) -> str:
    if study == "TCGA":
        return TCGA_CANCER_NAMES.get(cohort, cohort)
    return cohort.replace("_", " ")


def _cbio_url(study: str, cohort: str) -> str:
    if study != "TCGA":
        return ""
    study_id = "coadread" if cohort == "CRAD" else cohort.lower()
    return f"https://www.cbioportal.org/study/summary?id={study_id}_tcga_pan_can_atlas_2018"


def _load_drivers(path: Path) -> tuple[set[str], str]:
    table = pd.read_csv(path, sep="\t")
    if "Hugo Symbol" not in table:
        msg = f"driver reference missing 'Hugo Symbol': {path}"
        raise ValueError(msg)
    return set(table["Hugo Symbol"].astype(str)), _sha256(path)


def _hashed_file(path: Path) -> dict[str, object]:
    if not path.is_file():
        msg = f"missing provenance input: {path}"
        raise FileNotFoundError(msg)
    return {
        "path": path.as_posix(),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "modified_at_utc": datetime.fromtimestamp(
            path.stat().st_mtime,
            tz=UTC,
        )
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
    }


def _string_axis(index: pd.Index, *, label: str, axis: str) -> list[str]:
    """Return a non-null, unique axis as strings."""
    if index.hasnans:
        msg = f"{label}: null {axis} identifier"
        raise ValueError(msg)
    values = [str(value) for value in index]
    if any(not value for value in values):
        msg = f"{label}: empty {axis} identifier"
        raise ValueError(msg)
    if any(_has_control_characters(value) for value in values):
        msg = f"{label}: control character in {axis} identifier"
        raise ValueError(msg)
    if len(values) != len(set(values)):
        msg = f"{label}: duplicate {axis} identifier"
        raise ValueError(msg)
    return values


def _has_control_characters(value: str) -> bool:
    """Return whether an identifier contains a Unicode control character."""
    return any(unicodedata.category(character)[0] == "C" for character in value)


def _load_valid_count_matrix(path: Path, *, label: str) -> pd.DataFrame:
    """Load a release count matrix and fail closed on malformed scientific data."""
    counts = pd.read_csv(path, index_col=0)
    if counts.shape[0] == 0 or counts.shape[1] == 0:
        msg = f"{label}: count matrix must contain samples and features"
        raise ValueError(msg)
    samples = _string_axis(counts.index, label=label, axis="sample")
    features = _string_axis(counts.columns, label=label, axis="feature")
    invalid_features = [
        feature
        for feature in features
        if RAW_FEATURE_ID_PATTERN.fullmatch(feature) is None
    ]
    if invalid_features:
        msg = f"{label}: invalid _M/_N count-matrix feature IDs: {invalid_features[:5]}"
        raise ValueError(msg)
    try:
        numeric = counts.apply(pd.to_numeric, errors="raise")
        values = numeric.to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        msg = f"{label}: count matrix values must be numeric"
        raise ValueError(msg) from error
    if (
        not np.isfinite(values).all()
        or (values < 0).any()
        or not np.equal(values, np.floor(values)).all()
        or (values > np.iinfo(np.int64).max).any()
    ):
        msg = f"{label}: count matrix values must be finite nonnegative integers"
        raise ValueError(msg)
    numeric.index = pd.Index(samples)
    numeric.columns = pd.Index(features)
    return numeric.astype(np.int64)


def _load_valid_pmfs(path: Path, *, label: str) -> pd.DataFrame:
    """Load a PMF table while allowing only explicit sparse padding as missing."""
    frame = pd.read_csv(path, index_col=0)
    if frame.shape[0] == 0 or frame.shape[1] == 0:
        msg = f"{label}: PMF table must contain features and count columns"
        raise ValueError(msg)
    features = _string_axis(frame.index, label=label, axis="PMF feature")
    count_columns = _string_axis(frame.columns, label=label, axis="PMF count key")
    invalid_features = [
        feature
        for feature in features
        if RAW_FEATURE_ID_PATTERN.fullmatch(feature) is None
    ]
    if invalid_features:
        msg = f"{label}: invalid _M/_N PMF feature IDs: {invalid_features[:5]}"
        raise ValueError(msg)
    if any(
        NONNEGATIVE_INTEGER_PATTERN.fullmatch(column) is None
        for column in count_columns
    ):
        msg = f"{label}: PMF columns must be nonnegative integer count keys"
        raise ValueError(msg)
    parsed_count_columns = [int(column) for column in count_columns]
    if len(parsed_count_columns) != len(set(parsed_count_columns)):
        msg = f"{label}: duplicate integer PMF count key"
        raise ValueError(msg)
    try:
        numeric = frame.apply(pd.to_numeric, errors="raise")
        values = numeric.to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        msg = f"{label}: PMF probabilities must be numeric"
        raise ValueError(msg) from error
    populated = ~np.isnan(values)
    if not populated.any(axis=1).all():
        msg = f"{label}: every PMF row must contain at least one probability"
        raise ValueError(msg)
    probabilities = values[populated]
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        msg = f"{label}: PMF probabilities must be finite and nonnegative"
        raise ValueError(msg)
    row_mass = np.nansum(values, axis=1)
    if not np.allclose(
        row_mass,
        1.0,
        atol=PMF_SUM_TOLERANCE,
        rtol=0.0,
    ):
        invalid_rows = np.flatnonzero(
            ~np.isclose(
                row_mass,
                1.0,
                atol=PMF_SUM_TOLERANCE,
                rtol=0.0,
            ),
        )
        preview = [features[position] for position in invalid_rows[:5]]
        msg = f"{label}: PMF row probability mass differs from 1: {preview}"
        raise ValueError(msg)
    numeric.index = pd.Index(features)
    numeric.columns = pd.Index(parsed_count_columns)
    return numeric


def _read_mutsig_axis(
    path: Path,
    *,
    expected_length: int,
    label: str,
    axis: str,
    pattern: re.Pattern[str] | None,
) -> list[str]:
    """Read one exact, safe, unique MutSig label axis."""
    values = path.read_text(encoding="utf-8").splitlines()
    if len(values) != expected_length:
        msg = (
            f"{label}: MutSig {axis} axis length {len(values)} "
            f"does not match metadata {expected_length}"
        )
        raise ValueError(msg)
    invalid = [
        value
        for value in values
        if not value
        or _has_control_characters(value)
        or (pattern is not None and pattern.fullmatch(value) is None)
    ]
    if invalid:
        msg = f"{label}: unsafe MutSig {axis} identifiers: {invalid[:5]}"
        raise ValueError(msg)
    if len(values) != len(set(values)):
        msg = f"{label}: duplicate MutSig {axis} identifier"
        raise ValueError(msg)
    return values


def _load_valid_mutsig_inputs(
    mutsig_dir: Path,
    *,
    label: str,
) -> tuple[list[str], list[str], dict[str, int]]:
    """Validate MutSig axes, dimensions, and its little-endian float32 tensor."""
    meta_path = mutsig_dir / "persample_meta.txt"
    fields: dict[str, int] = {}
    for line in meta_path.read_text(encoding="utf-8").splitlines():
        pieces = line.split("\t")
        if (
            len(pieces) != 2
            or pieces[0] in fields
            or NONNEGATIVE_INTEGER_PATTERN.fullmatch(pieces[1]) is None
        ):
            msg = f"{label}: invalid MutSig metadata row: {line!r}"
            raise ValueError(msg)
        fields[pieces[0]] = int(pieces[1])
    expected_fields = {"ng", "np", "neff"}
    if set(fields) != expected_fields:
        msg = (
            f"{label}: MutSig metadata fields must be exactly "
            f"{sorted(expected_fields)}"
        )
        raise ValueError(msg)
    if fields["ng"] <= 0 or fields["np"] <= 0:
        msg = f"{label}: MutSig ng and np dimensions must be positive"
        raise ValueError(msg)
    if fields["neff"] != MUTSIG_EFFECT_COUNT:
        msg = f"{label}: MutSig neff must equal {MUTSIG_EFFECT_COUNT}"
        raise ValueError(msg)

    genes = _read_mutsig_axis(
        mutsig_dir / "persample_genes.txt",
        expected_length=fields["ng"],
        label=label,
        axis="gene",
        pattern=GENE_ID_PATTERN,
    )
    patients = _read_mutsig_axis(
        mutsig_dir / "persample_patients.txt",
        expected_length=fields["np"],
        label=label,
        axis="patient",
        pattern=None,
    )
    lambda_path = mutsig_dir / "persample_lambda.f32"
    expected_values = fields["ng"] * fields["np"] * fields["neff"]
    expected_bytes = expected_values * FLOAT32_BYTES
    actual_bytes = lambda_path.stat().st_size
    if actual_bytes != expected_bytes:
        msg = (
            f"{label}: MutSig lambda byte length {actual_bytes} does not match "
            f"metadata {expected_bytes}"
        )
        raise ValueError(msg)
    lambdas = np.memmap(
        lambda_path,
        dtype="<f4",
        mode="r",
        shape=(expected_values,),
    )
    try:
        for start in range(0, expected_values, VALIDATION_CHUNK_VALUES):
            values = lambdas[start : start + VALIDATION_CHUNK_VALUES]
            if not np.isfinite(values).all() or (values < 0).any():
                msg = f"{label}: MutSig lambda values must be finite and nonnegative"
                raise ValueError(msg)
    finally:
        del lambdas
    return genes, patients, fields


def _bmr_input_provenance(
    *,
    cohort_dir: Path,
    mutsig_dir: Path | None,
    count_features: list[str],
    count_samples: list[str],
) -> tuple[dict[str, object], dict[str, set[str]], dict[str, str]]:
    """Hash BMR inputs and reconstruct each provider's eligible feature set."""
    cbase_path = cohort_dir / "bmr_pmfs.csv"
    dig_path = cohort_dir / "bmr_pmfs.dig.csv"
    cbase_features = set(
        _load_valid_pmfs(cbase_path, label=f"{cohort_dir}/cbase").index,
    )
    dig_features = set(
        _load_valid_pmfs(dig_path, label=f"{cohort_dir}/dig").index,
    )
    if mutsig_dir is None:
        msg = f"missing MutSig provenance root for {cohort_dir}"
        raise FileNotFoundError(msg)
    mutsig_files = {name: mutsig_dir / name for name in MUTSIG_INPUT_NAMES}
    mutsig_gene_axis, mutsig_patient_axis, mutsig_dimensions = (
        _load_valid_mutsig_inputs(
            mutsig_dir,
            label=f"{cohort_dir}/mutsig",
        )
    )
    mutsig_genes = set(mutsig_gene_axis)
    mutsig_patients = set(mutsig_patient_axis)
    fallback_samples = sum(sample not in mutsig_patients for sample in count_samples)
    count_set = set(count_features)
    eligible = {
        "cbase": count_set & cbase_features,
        "dig": count_set & dig_features,
        "mutsig": {
            gene_effect
            for gene_effect in count_features
            if gene_effect in cbase_features
            or gene_effect.rsplit("_", 1)[0] in mutsig_genes
        },
    }
    mutsig_feature_origins = {
        gene_effect: (
            "mutsig_lambda"
            if gene_effect.rsplit("_", 1)[0] in mutsig_genes
            and gene_effect.rsplit("_", 1)[1] in {"M", "N"}
            else "cbase_fallback"
        )
        for gene_effect in eligible["mutsig"]
    }
    provenance = {
        "cbase": _hashed_file(cbase_path),
        "dig": _hashed_file(dig_path),
        "mutsig": {
            "directory": mutsig_dir.as_posix(),
            "files": {name: _hashed_file(path) for name, path in mutsig_files.items()},
            "dimensions": mutsig_dimensions,
            "sample_mapping": {
                "cohort_samples": len(count_samples),
                "matched_samples": len(count_samples) - fallback_samples,
                "cohort_mean_fallback_samples": fallback_samples,
                "fallback_policy": (
                    "samples absent from the MutSig patient axis receive the "
                    "gene-specific cohort-mean lambda"
                ),
            },
            "feature_origin_policy": (
                "MutSig per-sample lambda when the base gene is present on the "
                "MutSig gene axis; otherwise the cohort CBaSE PMF is used"
            ),
        },
    }
    return provenance, eligible, mutsig_feature_origins


def _count_ranked_features(
    counts: pd.DataFrame,
    eligible: set[str],
    bmr: str,
    *,
    k: int = TOP_K,
) -> list[str]:
    """Reproduce each provider's inference-time count ranking exactly."""
    totals = counts.sum(axis=0)
    if bmr == "mutsig":
        # The per-sample MutSig producer uses Series.sort_values directly.
        ordered = totals.sort_values(ascending=False, kind="quicksort").index
    else:
        # CBaSE/DIG preserve count-matrix column order for equal totals.
        ordered = sorted(
            counts.columns,
            key=lambda feature: totals[feature],
            reverse=True,
        )
    return [str(feature) for feature in ordered if str(feature) in eligible][:k]


def _require_exact_tested_features(
    actual: set[str],
    expected: list[str],
    *,
    label: str,
) -> None:
    """Require the exact provider-specific top-K feature set, not only its size."""
    expected_set = set(expected)
    if actual != expected_set:
        missing = sorted(expected_set - actual)[:5]
        extra = sorted(actual - expected_set)[:5]
        msg = (
            f"{label}: tested features are not the exact count-ranked K={TOP_K}; "
            f"missing={missing}, extra={extra}"
        )
        raise ValueError(msg)


def _public_baseline_provenance(manifest: dict[str, object]) -> dict[str, object]:
    """Retain reproducibility evidence without publishing machine-local paths."""
    source = manifest.get("provenance", {})
    discover = source.get("discover", {})
    return {
        "generated_at_utc": source.get("generated_at_utc"),
        "python": source.get("python"),
        "packages": source.get("packages"),
        "git": source.get("git"),
        "source_files": source.get("source_files"),
        "discover": {
            "version": discover.get("version"),
            "python_source_sha256": discover.get("python_source_sha256"),
        },
        "rscript": source.get("rscript"),
    }


def _iter_cohorts(
    sources: tuple[Source, ...],
) -> list[tuple[str, str, Path, Path | None]]:
    cohorts: list[tuple[str, str, Path, Path | None]] = []
    for source in sources:
        if not source.root.exists():
            continue
        cohort_dirs = [
            cohort_dir
            for cohort_dir in sorted(source.root.iterdir())
            if cohort_dir.is_dir() and (cohort_dir / "count_matrix.csv").exists()
        ]
        cohorts.extend(
            (
                source.study,
                cohort_dir.name,
                cohort_dir,
                source.mutsig_root / cohort_dir.name if source.mutsig_root else None,
            )
            for cohort_dir in cohort_dirs
        )
    return sorted(cohorts, key=lambda item: (item[0], item[1]))


def _cohort_payload(  # noqa: PLR0913 - explicit inputs form the cohort contract
    *,
    study: str,
    cohort: str,
    cohort_dir: Path,
    mutsig_dir: Path | None,
    baseline_root: Path,
    baseline_entry: dict[str, object],
    drivers: set[str],
) -> tuple[dict[str, object], dict[str, object]]:
    cohort_id = f"{study}__{cohort}"
    count_path = cohort_dir / "count_matrix.csv"
    counts = _load_valid_count_matrix(count_path, label=cohort_id)
    n_samples = len(counts)
    bmr_inputs, eligible_features, mutsig_feature_origins = _bmr_input_provenance(
        cohort_dir=cohort_dir,
        mutsig_dir=mutsig_dir,
        count_features=[str(column) for column in counts.columns],
        count_samples=[str(sample) for sample in counts.index],
    )

    models: dict[str, object] = {}
    model_feature_sets: dict[str, set[str]] = {}
    expected_model_features: dict[str, list[str]] = {}
    model_pair_sets: dict[str, set[tuple[str, str]]] = {}
    observed_genes: set[str] = set()
    for bmr in BMRS:
        path = cohort_dir / f"id_{bmr}" / "pairwise_interaction_results.csv"
        if not path.exists():
            msg = f"{cohort_id}: missing K=100 {bmr} results: {path}"
            raise FileNotFoundError(msg)
        model = dialect_payload(
            path,
            n_samples=n_samples,
            label=f"{cohort_id}/{bmr}",
            counts=counts,
        )
        models[bmr] = model
        model_pairs = {_pair_key(str(row[0]), str(row[1])) for row in model["rows"]}
        model_pair_sets[bmr] = model_pairs
        model_feature_sets[bmr] = {gene for pair in model_pairs for gene in pair}
        expected_model_features[bmr] = _count_ranked_features(
            counts,
            eligible_features[bmr],
            bmr,
        )
        _require_exact_tested_features(
            model_feature_sets[bmr],
            expected_model_features[bmr],
            label=f"{cohort_id}/{bmr}",
        )
        for row in model["rows"]:
            observed_genes.update((_base_gene(row[0]), _base_gene(row[1])))

    baseline_path = (
        baseline_root / study / cohort / "comparison_pairwise_interaction_results.csv"
    )
    if not baseline_path.exists():
        msg = f"{cohort_id}: missing complete K=100 baseline results: {baseline_path}"
        raise FileNotFoundError(msg)
    baselines = baseline_payload(baseline_path, label=f"{cohort_id}/baselines")
    baseline_metadata = baselines["source"]["metadata"]
    if baseline_metadata.get("source_gene_k") != TOP_K:
        msg = f"{cohort_id}: baseline metadata is not K={TOP_K}"
        raise ValueError(msg)
    if baseline_metadata.get("cohort", {}).get("id") != cohort_id:
        msg = f"{cohort_id}: baseline metadata cohort ID mismatch"
        raise ValueError(msg)
    count_hash = _sha256(count_path)
    if baseline_metadata.get("input", {}).get("sha256") != count_hash:
        msg = f"{cohort_id}: baseline input hash does not match the count matrix"
        raise ValueError(msg)
    totals = counts.sum(axis=0)
    expected_baseline_features = [
        str(feature)
        for feature in sorted(
            counts.columns,
            key=lambda feature: totals[feature],
            reverse=True,
        )[: min(TOP_K, len(counts.columns))]
    ]
    if baseline_metadata.get("top_features") != expected_baseline_features:
        msg = f"{cohort_id}: baseline top-feature selection is not K={TOP_K}"
        raise ValueError(msg)
    metadata_path = baseline_path.with_name("metadata.json")
    if baseline_entry.get("id") != cohort_id:
        msg = f"{cohort_id}: baseline root-manifest cohort mismatch"
        raise ValueError(msg)
    root_comparison = baseline_entry.get("artifacts", {}).get("comparison", {})
    root_metadata = baseline_entry.get("artifacts", {}).get("metadata", {})
    if root_comparison.get("sha256") != baselines["source"]["sha256"]:
        msg = f"{cohort_id}: baseline artifact is not linked by the root manifest"
        raise ValueError(msg)
    if root_metadata.get("sha256") != _sha256(metadata_path):
        msg = f"{cohort_id}: baseline metadata is not linked by the root manifest"
        raise ValueError(msg)
    for field in ("top_features", "method_coverage", "input"):
        if baseline_entry.get(field) != baseline_metadata.get(field):
            msg = f"{cohort_id}: baseline root/local {field} mismatch"
            raise ValueError(msg)

    baseline_pairs = {_pair_key(str(row[0]), str(row[1])) for row in baselines["rows"]}
    baseline_features = {gene for pair in baseline_pairs for gene in pair}
    common_dialect_features = set.intersection(*model_feature_sets.values())
    union_dialect_features = set.union(*model_feature_sets.values())
    common_dialect_pairs = set.intersection(*model_pair_sets.values())
    mutsig_origin_lists = {
        origin: [
            feature
            for feature in expected_model_features["mutsig"]
            if mutsig_feature_origins[feature] == origin
        ]
        for origin in ("mutsig_lambda", "cbase_fallback")
    }
    mutsig_fallback_features = set(mutsig_origin_lists["cbase_fallback"])
    mutsig_fallback_pairs = {
        pair
        for pair in model_pair_sets["mutsig"]
        if mutsig_fallback_features.intersection(pair)
    }
    testing_universes = {
        "policy": (
            "top count-ranked features available to each BMR; consensus uses only "
            "exact pairs tested by all three"
        ),
        "models": {
            bmr: {
                "features": expected_model_features[bmr],
                "features_sha256": _sequence_sha256(expected_model_features[bmr]),
                **(
                    {
                        "origins": mutsig_origin_lists,
                        "origin_summary": {
                            origin: len(features)
                            for origin, features in mutsig_origin_lists.items()
                        },
                    }
                    if bmr == "mutsig"
                    else {}
                ),
            }
            for bmr in BMRS
        },
        "baseline": {"features": sorted(baseline_features)},
        "summary": {
            "common_dialect_features": len(common_dialect_features),
            "union_dialect_features": len(union_dialect_features),
            "common_dialect_pairs": len(common_dialect_pairs),
            "baseline_pairs_shared_with_all_dialect": len(
                baseline_pairs & common_dialect_pairs,
            ),
            "mutsig_pairs_with_cbase_fallback": len(mutsig_fallback_pairs),
            "common_dialect_pairs_with_mutsig_fallback": len(
                common_dialect_pairs & mutsig_fallback_pairs,
            ),
        },
    }

    payload = {
        "id": cohort_id,
        "drivers": sorted(observed_genes & drivers),
        "models": models,
        "baselines": baselines,
        "testing_universes": testing_universes,
        "provenance": {
            "count_matrix": {
                **_hashed_file(count_path),
            },
            "bmr_inputs": bmr_inputs,
        },
    }
    index_record = {
        "id": cohort_id,
        "study": study,
        "cohort": cohort,
        "cancer": _cancer_name(study, cohort),
        "n_samples": n_samples,
        "median_mutations": float(counts.sum(axis=1).median()),
        "cbio": _cbio_url(study, cohort),
        "model_summaries": {bmr: models[bmr]["summary"] for bmr in BMRS},
        "baseline_summary": baselines["summary"],
        "testing_universe": {
            "model_features": {bmr: len(model_feature_sets[bmr]) for bmr in BMRS},
            "eligible_model_features": {
                bmr: len(eligible_features[bmr]) for bmr in BMRS
            },
            "expected_tested_features": {
                bmr: len(expected_model_features[bmr]) for bmr in BMRS
            },
            "model_feature_sha256": {
                bmr: _sequence_sha256(expected_model_features[bmr]) for bmr in BMRS
            },
            "mutsig_feature_origins": {
                origin: len(features)
                for origin, features in mutsig_origin_lists.items()
            },
            "mutsig_cbase_fallback_sha256": _sequence_sha256(
                mutsig_origin_lists["cbase_fallback"],
            ),
            "baseline_features": len(baseline_features),
            **testing_universes["summary"],
        },
    }
    return payload, index_record


def _build_release_tree(  # noqa: PLR0913 - explicit inputs form the release contract
    *,
    out: Path,
    baseline_root: Path,
    sources: tuple[Source, ...] = DEFAULT_SOURCES,
    drivers_path: Path = Path("data/references/OncoKB_Cancer_Gene_List.tsv"),
    release_id: str = RELEASE_ID,
    generated_at: str | None = None,
    require_committed_sources: bool = True,
) -> dict[str, object]:
    """Build and validate one Atlas release inside an isolated staging tree."""
    out.mkdir(parents=True, exist_ok=True)
    if require_committed_sources:
        _require_committed_release_sources(RELEASE_SOURCE_FILES)
    baseline_manifest_path = baseline_root / "manifest.json"
    if not baseline_manifest_path.exists():
        msg = f"missing immutable baseline manifest: {baseline_manifest_path}"
        raise FileNotFoundError(msg)
    baseline_manifest = json.loads(baseline_manifest_path.read_text())
    if baseline_manifest.get("source_gene_k") != TOP_K:
        msg = f"baseline release is not K={TOP_K}: {baseline_manifest_path}"
        raise ValueError(msg)
    discover_provenance = baseline_manifest.get("provenance", {}).get(
        "discover",
        {},
    )
    if (
        discover_provenance.get("version") != DISCOVER_VERSION
        or discover_provenance.get("python_source_sha256") != DISCOVER_SOURCE_SHA256
    ):
        msg = "baseline DISCOVER source does not match the pinned 0.9.6 release"
        raise ValueError(msg)
    baseline_source_hashes = baseline_manifest.get("provenance", {}).get(
        "source_files",
        {},
    )
    if require_committed_sources and not baseline_source_hashes:
        msg = "baseline release does not publish its implementation source hashes"
        raise ValueError(msg)
    baseline_source_paths: list[Path] = []
    for path_string, expected_hash in baseline_source_hashes.items():
        source_path = Path(path_string)
        if source_path.is_absolute() or ".." in source_path.parts:
            msg = f"invalid baseline source path: {path_string}"
            raise ValueError(msg)
        if not source_path.is_file() or _sha256(source_path) != expected_hash:
            msg = f"baseline source snapshot mismatch: {path_string}"
            raise ValueError(msg)
        baseline_source_paths.append(source_path)
    if require_committed_sources:
        _require_committed_release_sources(tuple(baseline_source_paths))
    drivers, driver_sha = _load_drivers(drivers_path)
    cohorts = _iter_cohorts(sources)
    if not cohorts:
        msg = "no cohort count matrices found"
        raise FileNotFoundError(msg)
    cohort_ids = tuple(f"{study}__{cohort}" for study, cohort, _, _ in cohorts)
    if sources == DEFAULT_SOURCES and cohort_ids != EXPECTED_COHORT_IDS:
        missing = sorted(set(EXPECTED_COHORT_IDS) - set(cohort_ids))
        extra = sorted(set(cohort_ids) - set(EXPECTED_COHORT_IDS))
        msg = (
            "canonical Atlas cohort identity lock failed; "
            f"missing={missing}, extra={extra}"
        )
        raise ValueError(msg)
    if baseline_manifest.get("cohort_count") != len(cohorts):
        msg = (
            "baseline manifest cohort count does not match Atlas inputs: "
            f"{baseline_manifest.get('cohort_count')} != {len(cohorts)}"
        )
        raise ValueError(msg)
    baseline_entries = {
        str(entry.get("id")): entry
        for entry in baseline_manifest.get("cohorts", [])
        if isinstance(entry, dict)
    }
    expected_ids = {f"{study}__{cohort}" for study, cohort, _, _ in cohorts}
    if set(baseline_entries) != expected_ids:
        msg = "baseline root manifest does not exactly cover the Atlas cohort IDs"
        raise ValueError(msg)

    index_records: list[dict[str, object]] = []
    for study, cohort, cohort_dir, mutsig_dir in cohorts:
        payload, record = _cohort_payload(
            study=study,
            cohort=cohort,
            cohort_dir=cohort_dir,
            mutsig_dir=mutsig_dir,
            baseline_root=baseline_root,
            baseline_entry=baseline_entries[f"{study}__{cohort}"],
            drivers=drivers,
        )
        relative = Path("cohorts") / f"{record['id']}.json"
        output_path = out / relative
        _write_json(output_path, payload)
        record.update(
            {
                "data_file": relative.as_posix(),
                "data_sha256": _sha256(output_path),
                "data_bytes": output_path.stat().st_size,
            },
        )
        index_records.append(record)

    index = {"release_id": release_id, "cohorts": index_records}
    index_path = out / "index.json"
    _write_json(index_path, index)
    readme_path = out / "README.md"
    _write_release_readme(readme_path, release_id)
    generated = generated_at or (
        datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )
    studies = Counter(record["study"] for record in index_records)
    generator_path = Path("analysis/build_atlas_data.py")
    if require_committed_sources:
        _require_committed_release_sources(RELEASE_SOURCE_FILES)
    source_snapshot = {path.as_posix(): _sha256(path) for path in RELEASE_SOURCE_FILES}
    manifest = {
        "release_id": release_id,
        "schema_version": SCHEMA_VERSION,
        "immutable": True,
        "generated_at": generated,
        "title": "DIALECT Atlas complete K=100 release",
        "coverage": {
            "cohorts": len(index_records),
            "cohort_ids_sha256": _sequence_sha256(
                tuple(record["id"] for record in index_records),
            ),
            "studies": dict(sorted(studies.items())),
            "samples": sum(record["n_samples"] for record in index_records),
            "dialect_tables": len(index_records) * len(BMRS),
            "baseline_tables": len(index_records),
            "mutsig_cbase_fallback_feature_instances": sum(
                record["testing_universe"]["mutsig_feature_origins"]["cbase_fallback"]
                for record in index_records
            ),
            "mutsig_pair_rows_with_cbase_fallback": sum(
                record["testing_universe"]["mutsig_pairs_with_cbase_fallback"]
                for record in index_records
            ),
        },
        "analysis": {
            "top_k_event_features": TOP_K,
            "p_value": "chi2.sf(max(lrt, 0), df=1)",
            "multiple_testing": (
                "Benjamini-Hochberg across all evaluated DIALECT pairs per cohort "
                "and BMR; ME and CO share one family"
            ),
            "fdr_threshold": FDR,
            "fdr_operator": "<",
            "direction": "rho < 0: ME; rho > 0: CO; rho = 0: neutral",
            "ranking": {"ME": "rho ascending", "CO": "raw LRT descending"},
            "epsilon_filter": False,
            "same_base_pair_policy": (
                "retained in testing family; hidden only in the web presentation"
            ),
            "feature_universe_policy": (
                "each BMR uses the top K count-ranked features for which that BMR "
                "provides a PMF; baseline methods use their own count-ranked K; "
                "cross-BMR consensus contains only exact pairs tested by all three"
            ),
            "feature_tie_policy": {
                "cbase_dig": "count-matrix column order for equal totals",
                "mutsig": (
                    "historical pandas Series.sort_values descending quicksort; "
                    "the ordered selected list and digest are published per cohort"
                ),
                "validation_pandas_version": pd.__version__,
            },
            "negative_lrt_policy": (
                "raw value preserved; clamped to zero only for p-value computation "
                "and evidence display"
            ),
            "default_consensus": (
                "exact gene-effect pair with the same direction in CBaSE, DIG, and "
                "MutSigCV2; ordered by worst directional rank percentile then "
                "median percentile"
            ),
            "strict_consensus": "q < 0.01 in all three BMRs",
            "unsupported_sample_policy": (
                "raw fitted tau mass is preserved; pair-level effective_n and "
                "excluded_samples report samples with no background support; "
                "these fits may be biased and are not renormalized"
            ),
            "mutsig_feature_fallback_policy": (
                "MutSig uses per-sample lambda only for gene-effects whose base "
                "gene is on its gene axis; otherwise the historical producer "
                "uses the CBaSE PMF. Fallback features are published per cohort "
                "and excluded from the default all-three consensus."
            ),
        },
        "bmrs": [{"id": bmr, **BMR_METADATA[bmr]} for bmr in BMRS],
        "methods": {
            "dialect": {
                "directions": ["ME", "CO"],
                "multiple_testing_family": "unified ME+CO",
            },
            "fisher": {
                "directions": ["ME", "CO"],
                "multiple_testing_family": "separate by direction",
            },
            "discover": {
                "directions": ["ME", "CO"],
                "version": baseline_manifest.get("provenance", {})
                .get("discover", {})
                .get("version"),
                "upstream": {
                    "repository": "https://github.com/NKI-CCB/DISCOVER",
                    "tag": "py_v0.9.6",
                    "commit": DISCOVER_COMMIT,
                    "python_source_sha256": DISCOVER_SOURCE_SHA256,
                },
                "multiple_testing_family": "separate by direction",
            },
            "megsa": {
                "directions": ["ME"],
                "call_rule": "p < 0.001",
            },
            "wesme_wesco": {
                "directions": ["ME", "CO"],
                "multiple_testing_family": "separate by direction",
                "seeded": True,
            },
        },
        "provenance": {
            "dialect_repository": "https://github.com/raphael-group/dialect",
            "release_assembly_commit": _git_revision(),
            "inference_run_commit": None,
            "inference_run_commit_note": (
                "not recorded by the historical cohort pipeline; exact result and "
                "input hashes are provided, and source_snapshot records the "
                "implementation present when this release was assembled"
            ),
            "generator": generator_path.as_posix(),
            "generator_sha256": _sha256(generator_path),
            "source_snapshot": source_snapshot,
            "driver_reference": drivers_path.as_posix(),
            "driver_reference_sha256": driver_sha,
            "baseline_release": {
                "path": baseline_manifest_path.as_posix(),
                "sha256": _sha256(baseline_manifest_path),
                "release_id": baseline_manifest.get("release_id"),
                "release_seed": baseline_manifest.get("release_seed"),
                "provenance": _public_baseline_provenance(baseline_manifest),
            },
        },
        "index_file": "index.json",
        "index_sha256": _sha256(index_path),
        "index_bytes": index_path.stat().st_size,
        "readme_file": "README.md",
        "readme_sha256": _sha256(readme_path),
        "readme_bytes": readme_path.stat().st_size,
    }
    _write_json(out / "manifest.json", manifest)
    return manifest


def build_release(  # noqa: PLR0913 - explicit inputs form the release contract
    *,
    out: Path,
    baseline_root: Path,
    sources: tuple[Source, ...] = DEFAULT_SOURCES,
    drivers_path: Path = Path("data/references/OncoKB_Cancer_Gene_List.tsv"),
    release_id: str = RELEASE_ID,
    generated_at: str | None = None,
    require_committed_sources: bool = True,
) -> dict[str, object]:
    """Build an immutable release atomically, refusing every overwrite."""
    if out.exists():
        msg = f"immutable release target already exists: {out}"
        raise FileExistsError(msg)
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=out.parent))
    try:
        manifest = _build_release_tree(
            out=staging,
            baseline_root=baseline_root,
            sources=sources,
            drivers_path=drivers_path,
            release_id=release_id,
            generated_at=generated_at,
            require_committed_sources=require_committed_sources,
        )
        staging.rename(out)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def main() -> None:
    """Parse command-line arguments and build the release."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=Path("output/atlas_baselines/k100"),
    )
    parser.add_argument("--release-id", default=RELEASE_ID)
    parser.add_argument("--generated-at")
    args = parser.parse_args()
    manifest = build_release(
        out=args.out,
        baseline_root=args.baseline_root,
        release_id=args.release_id,
        generated_at=args.generated_at,
    )
    coverage = manifest["coverage"]
    print(
        f"wrote immutable release {manifest['release_id']} to {args.out}: "
        f"{coverage['cohorts']} cohorts, {coverage['dialect_tables']} DIALECT tables, "
        f"{coverage['baseline_tables']} baseline tables",
    )


if __name__ == "__main__":
    main()
