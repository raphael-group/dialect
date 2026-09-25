r"""Assemble the immutable K=500 DIALECT Atlas release from the focused revision grid.

The K=500 release publishes the complete tested family of the manuscript revision:
32 TCGA cohorts x 3 background models (MutSigCV2 primary, CBaSE continuity, DIG
sensitivity), every one of the ~124,700 tested pairs per cohort and model, and the
Fisher / DISCOVER / MEGSA / WeSME-WeSCO comparison methods recomputed on the same
pair family. MSK cohorts were not fitted at K=500 and remain in the K=100 release.

Statistical contract (the frozen revision reporting rule, not re-derived here)
------------------------------------------------------------------------------
* One complete within-cohort family per provider: every unordered pair of the frozen
  500-feature axis, same-gene missense/nonsense pairs excluded before fitting.
* ``p`` is the chi-square(1) profile-LRT survival probability for pairs whose effect
  is identifiable (``full-affine-rank``); otherwise ``p = 1``.
* Benjamini-Yekutieli is the primary adjustment and Benjamini-Hochberg a nominal
  sensitivity; threshold decisions compare natural-log q-values, inclusively
  (``log q <= log t``). MutSigCV2 is the only inferential provider; CBaSE and DIG
  crossings are descriptive.
* Direction is the provider's Marshall-Olkin rho sign, reported only after a
  nondirectional rejection; rejections without a defined sign stay in the family but
  are excluded from ME/CO lists.

Every p, q, log-q, rho, and direction value is read verbatim from the sealed
postprocess stage that produced the manuscript tables, and the per-cohort crossing
counts are asserted equal to the published Table S5 before anything is written.

Encoding
--------
Each cohort is a directory ``cohorts/<study>__<cohort>/`` holding ``cohort.json`` and
gzip-compressed columnar binary tables. A table stores each column contiguously,
little-endian, in the order listed by ``cohort.json``; float64 columns come first,
then uint32, uint16, and uint8, so every column starts at a multiple of its item
size. IEEE-754 NaN encodes null. Pairs are stored once (``pairs``) as indices into the
published feature axis; model and baseline tables are row-aligned to it.

Usage::

    python -m analysis.build_atlas_data --k 500 \
      --out atlas/public/data/releases/k500-2026-09-25 \
      --release-id k500-2026-09-25 \
      --generated-at 2026-09-25T00:00:00Z
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import shutil
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis import postprocess_tcga_revision_focused as postprocess

SCHEMA_VERSION = "3.0.0"
TOP_K = 500
STUDY = "TCGA"
COHORT_COUNT = 32
BMRS = ("mutsig", "cbase", "dig")
BMR_METADATA = {
    "mutsig": {"label": "MutSigCV2", "role": "primary"},
    "cbase": {"label": "CBaSE", "role": "continuity"},
    "dig": {"label": "DIG", "role": "sensitivity"},
}
ADJUSTMENTS = {"by": "benjamini-yekutieli", "bh": "benjamini-hochberg"}
DECISION_THRESHOLDS = (0.001, 0.005, 0.01, 0.05)
DIRECTIONS = ("unavailable", "ME", "CO", "neutral")
DIRECTION_CODES = {label: code for code, label in enumerate(DIRECTIONS)}
COMPLETION_CONTRACT = "focused-32x3-k500-completion-v1"
POSTPROCESS_CONTRACT = "focused-32x3-provider-inference-v4"
RULE_CONTRACT = "focused-global-reporting-rule-v4"
BASELINE_RELEASE_ID = "dialect-atlas-baselines-k500"
GZIP_LEVEL = 9
MAX_UINT16 = np.iinfo(np.uint16).max

DEFAULT_PROVIDER_ROOT = Path("output/tcga_revision_focused_providers_v1")
DEFAULT_RUN_ROOT = Path("output/tcga_revision_focused_k500_v1")
DEFAULT_POSTPROCESS_ROOT = Path("output/tcga_revision_focused_postprocess_v5")
DEFAULT_RULE_PATH = Path("output/tcga_revision_focused_reporting_rule_v5.json")
DEFAULT_CONFIRMATION_SUMMARY = Path(
    "output/tcga_revision_focused_calibration_confirmation_v1/confirmation_summary.json",
)
DEFAULT_REPORT_ROOT = Path("output/tcga_revision_focused_report_v17")
DEFAULT_RELEASE_RECEIPT = Path(
    "output/tcga_revision_focused_release_repro_a_v3/dialect-focused-k500-v1.receipt.json",
)
DEFAULT_BASELINE_ROOT = Path("output/atlas_baselines/k500")
DEFAULT_DRIVERS = Path("data/references/OncoKB_Cancer_Gene_List.tsv")

RELEASE_SOURCE_FILES = (
    Path("analysis/build_atlas_data.py"),
    Path("analysis/build_atlas_data_k500.py"),
    Path("analysis/build_atlas_baselines.py"),
    Path("analysis/postprocess_tcga_revision_focused.py"),
)

PAIR_COLUMNS = (
    ("a", "uint16"),
    ("b", "uint16"),
    ("observed_both", "uint16"),
    ("observed_a_only", "uint16"),
    ("observed_b_only", "uint16"),
    ("observed_neither", "uint16"),
)
# (published name, dtype, source) -- source "inference:<suffix>" reads
# ``<provider>_<suffix>`` from provider_inference.csv; "task:<column>" reads the
# fitted pairwise table; "derived" is computed by this module.
MODEL_COLUMNS = (
    ("lrt", "float64", "inference:likelihood_ratio"),
    ("log_p", "float64", "inference:log_p_value"),
    ("p", "float64", "inference:p_value"),
    ("log_by_q", "float64", "inference:log_by_q_value"),
    ("by_q", "float64", "inference:by_q_value"),
    ("log_bh_q", "float64", "inference:log_bh_q_value"),
    ("bh_q", "float64", "inference:bh_q_value"),
    ("rho", "float64", "inference:rho"),
    ("tau00", "float64", "task:Tau_00"),
    ("tau10", "float64", "task:Tau_10"),
    ("tau01", "float64", "task:Tau_01"),
    ("tau11", "float64", "task:Tau_11"),
    ("log_odds_ratio", "float64", "task:Log Odds Ratio"),
    ("wald", "float64", "task:Wald Statistic"),
    ("null_log_likelihood", "float64", "task:Null Log Likelihood"),
    ("alternative_log_likelihood", "float64", "task:Alternative Log Likelihood"),
    ("fit_last_ll_gain", "float64", "inference:fit_last_ll_gain"),
    ("fit_fixed_point_residual", "float64", "inference:fit_fixed_point_residual"),
    ("fit_kkt_residual", "float64", "inference:fit_kkt_residual"),
    ("rank", "uint32", "derived"),
    ("fit_iterations", "uint32", "inference:fit_iterations"),
    ("direction", "uint8", "derived"),
    ("identifiability", "uint8", "derived"),
    ("effect_reportable", "uint8", "inference:effect_reportable"),
    ("fit_converged", "uint8", "inference:fit_converged"),
    ("by_decisions", "uint8", "derived"),
    ("bh_decisions", "uint8", "derived"),
)
BASELINE_COLUMNS = (
    ("fisher_me_p", "Fisher's ME P-Val"),
    ("fisher_co_p", "Fisher's CO P-Val"),
    ("fisher_me_q", "Fisher's ME Q-Val"),
    ("fisher_co_q", "Fisher's CO Q-Val"),
    ("discover_me_p", "Discover ME P-Val"),
    ("discover_co_p", "Discover CO P-Val"),
    ("discover_me_q", "Discover ME Q-Val"),
    ("discover_co_q", "Discover CO Q-Val"),
    ("megsa_lrt", "MEGSA S-Score (LRT)"),
    ("megsa_p", "MEGSA P-Val"),
    ("megsa_q", "MEGSA Q-Val"),
    ("wesme_p", "WeSME P-Val"),
    ("wesco_p", "WeSCO P-Val"),
    ("wesme_q", "WeSME Q-Val"),
    ("wesco_q", "WeSCO Q-Val"),
)
DTYPE_ORDER = ("float64", "uint32", "uint16", "uint8")
NUMPY_DTYPES = {
    "float64": np.dtype("<f8"),
    "uint32": np.dtype("<u4"),
    "uint16": np.dtype("<u2"),
    "uint8": np.dtype("u1"),
}
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


@dataclass(frozen=True)
class FocusedInputs:
    """Every locked input root of the focused K=500 release."""

    provider_root: Path = DEFAULT_PROVIDER_ROOT
    run_root: Path = DEFAULT_RUN_ROOT
    postprocess_root: Path = DEFAULT_POSTPROCESS_ROOT
    rule_path: Path = DEFAULT_RULE_PATH
    confirmation_summary: Path = DEFAULT_CONFIRMATION_SUMMARY
    report_root: Path | None = DEFAULT_REPORT_ROOT
    release_receipt: Path | None = DEFAULT_RELEASE_RECEIPT
    baseline_root: Path = DEFAULT_BASELINE_ROOT
    drivers_path: Path = DEFAULT_DRIVERS
    expected_cohorts: int = COHORT_COUNT


# --------------------------------------------------------------------------- #
#                                   helpers                                   #
# --------------------------------------------------------------------------- #
def sha256_bytes(payload: bytes) -> str:
    """Return the SHA-256 hex digest of bytes."""
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        msg = f"missing provenance input: {path}"
        raise FileNotFoundError(msg)
    return {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        msg = f"missing required JSON input: {path}"
        raise FileNotFoundError(msg)
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
        encoding="utf-8",
    )


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _git(arguments: list[str]) -> bytes:
    return subprocess.run(["git", *arguments], check=True, capture_output=True).stdout


def _require_committed_sources(paths: tuple[Path, ...]) -> None:
    """Fail unless every release source is tracked and byte-identical to HEAD."""
    for path in paths:
        try:
            committed = _git(["show", f"HEAD:{path.as_posix()}"])
        except (OSError, subprocess.CalledProcessError) as error:
            msg = f"release source is not committed at HEAD: {path}"
            raise RuntimeError(msg) from error
        if not path.is_file() or path.read_bytes() != committed:
            msg = f"release source differs from HEAD: {path}"
            raise RuntimeError(msg)


def _git_revision() -> str:
    try:
        return _git(["rev-parse", "HEAD"]).decode().strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def decision_bits(log_q: np.ndarray) -> np.ndarray:
    """Encode inclusive ``log q <= log t`` crossings (bit i: DECISION_THRESHOLDS[i])."""
    bits = np.zeros(len(log_q), dtype=np.uint8)
    for position, threshold in enumerate(DECISION_THRESHOLDS):
        bits |= (log_q <= math.log(threshold)).astype(np.uint8) << position
    return bits


def direction_ranks(
    direction: np.ndarray,
    log_q: np.ndarray,
    log_p: np.ndarray,
    rho: np.ndarray,
) -> np.ndarray:
    """Rank each ME/CO row within its direction exactly as the revision report does.

    Order: primary log q ascending, log p ascending, |rho| descending, then the
    canonical pair order. Rows without an ME/CO direction receive rank 0.
    """
    ranks = np.zeros(len(direction), dtype=np.uint32)
    position = np.arange(len(direction))
    for label in ("ME", "CO"):
        members = position[direction == DIRECTION_CODES[label]]
        order = np.lexsort(
            (members, -np.abs(rho[members]), log_p[members], log_q[members]),
        )
        ranks[members[order]] = np.arange(1, len(members) + 1, dtype=np.uint32)
    return ranks


def encode_table(
    columns: dict[str, np.ndarray],
    spec: tuple[tuple[str, str], ...],
) -> tuple[bytes, list[dict[str, Any]]]:
    """Pack named columns into one aligned little-endian columnar byte string."""
    order = sorted(
        range(len(spec)),
        key=lambda index: (DTYPE_ORDER.index(spec[index][1]), index),
    )
    rows = None
    chunks: list[bytes] = []
    layout: list[dict[str, Any]] = []
    offset = 0
    for index in order:
        name, dtype = spec[index]
        array = np.ascontiguousarray(columns[name], dtype=NUMPY_DTYPES[dtype])
        if array.ndim != 1:
            msg = f"column {name} must be one-dimensional"
            raise ValueError(msg)
        if rows is None:
            rows = len(array)
        elif len(array) != rows:
            msg = f"column {name} has {len(array)} rows, expected {rows}"
            raise ValueError(msg)
        if offset % NUMPY_DTYPES[dtype].itemsize:
            msg = f"column {name} would be misaligned at byte {offset}"
            raise ValueError(msg)
        payload = array.tobytes()
        layout.append({"name": name, "dtype": dtype, "offset": offset})
        chunks.append(payload)
        offset += len(payload)
    return b"".join(chunks), layout


def decode_table(
    raw: bytes,
    rows: int,
    layout: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    """Inverse of :func:`encode_table` (used by verification and tests)."""
    columns = {}
    for column in layout:
        dtype = NUMPY_DTYPES[column["dtype"]]
        start = column["offset"]
        stop = start + rows * dtype.itemsize
        if stop > len(raw):
            msg = f"column {column['name']} overruns its table"
            raise ValueError(msg)
        columns[column["name"]] = np.frombuffer(raw[start:stop], dtype=dtype)
    return columns


def _write_table(
    directory: Path,
    name: str,
    columns: dict[str, np.ndarray],
    spec: tuple[tuple[str, str], ...],
) -> dict[str, Any]:
    raw, layout = encode_table(columns, spec)
    compressed = gzip.compress(raw, compresslevel=GZIP_LEVEL, mtime=0)
    filename = f"{name}.bin.gz"
    (directory / filename).write_bytes(compressed)
    return {
        "file": filename,
        "rows": len(next(iter(columns.values()))),
        "columns": layout,
        "bytes": len(compressed),
        "sha256": sha256_bytes(compressed),
        "raw_bytes": len(raw),
        "raw_sha256": sha256_bytes(raw),
    }


# --------------------------------------------------------------------------- #
#                               input validation                              #
# --------------------------------------------------------------------------- #
def load_rule(inputs: FocusedInputs) -> dict[str, Any]:
    """Load the frozen reporting rule and require its calibration gate and bindings."""
    rule = _read_json(inputs.rule_path)
    expected = {
        "contract": RULE_CONTRACT,
        "inference_status": "reportable",
        "primary_provider": "mutsig",
        "primary_adjustment": ADJUSTMENTS["by"],
        "sensitivity_adjustment": ADJUSTMENTS["bh"],
        "primary_q_threshold": 0.01,
        "sensitivity_q_threshold": 0.01,
        "threshold_comparison": "inclusive-less-than-or-equal",
        "multiplicity": "provider-specific-complete-within-cohort-family",
        "effective_p_policy": "chi-square-one-df-for-full-affine-rank-otherwise-p-one",
        "direction": "primary-provider-rho-sign-after-nondirectional-rejection",
        "direction_unavailable": (
            "retain-nondirectional-rejection-exclude-from-me-co-lists"
        ),
        "provider_overlap": "descriptive-only-not-an-inferential-vote",
        "thresholds_selected_from_observed_pairs": False,
    }
    drift = {
        key: rule.get(key) for key, value in expected.items() if rule.get(key) != value
    }
    if drift:
        msg = f"reporting rule differs from the frozen revision contract: {drift}"
        raise ValueError(msg)
    if rule.get("calibration_gate", {}).get("overall_gate_pass") is not True:
        msg = "reporting rule calibration gate did not pass; inference is withheld"
        raise RuntimeError(msg)
    postprocess_manifest = inputs.postprocess_root / "postprocess_manifest.json"
    bindings = {
        "postprocess_manifest_sha256": postprocess_manifest,
        "calibration_confirmation_summary_sha256": inputs.confirmation_summary,
    }
    for key, path in bindings.items():
        if rule.get(key) != sha256_file(path):
            msg = f"reporting rule {key} does not match {path}"
            raise ValueError(msg)
    confirmation = _read_json(inputs.confirmation_summary)
    if confirmation.get("composite_overall_gate_pass") is not True:
        msg = "calibration confirmation gate did not pass; inference is withheld"
        raise RuntimeError(msg)
    return rule


def load_completion(inputs: FocusedInputs) -> list[str]:
    """Verify the completed grid's manifest chain down to every task output hash."""
    completion = _read_json(inputs.run_root / "completion_manifest.json")
    if completion.get("contract") != COMPLETION_CONTRACT:
        msg = "focused completion manifest has an unexpected contract"
        raise ValueError(msg)
    cohorts = [str(cohort) for cohort in completion.get("cohorts", [])]
    if (
        len(cohorts) != inputs.expected_cohorts
        or len(set(cohorts)) != len(cohorts)
        or completion.get("task_count") != len(cohorts) * len(BMRS)
    ):
        msg = "focused completion manifest does not cover the locked grid"
        raise ValueError(msg)
    seen = set()
    for task in completion.get("tasks", []):
        cohort, provider = task.get("cohort"), task.get("provider")
        record = task.get("manifest", {})
        manifest_path = inputs.run_root / str(record.get("path"))
        if sha256_file(manifest_path) != record.get("sha256"):
            msg = f"task manifest changed since completion: {cohort}/{provider}"
            raise ValueError(msg)
        manifest = _read_json(manifest_path)
        if manifest.get("top_k") != TOP_K:
            msg = f"task is not K={TOP_K}: {cohort}/{provider}"
            raise ValueError(msg)
        for name, output in manifest.get("outputs", {}).items():
            path = manifest_path.parent / name
            if sha256_file(path) != output.get("sha256"):
                msg = f"task output changed since completion: {path}"
                raise ValueError(msg)
        seen.add((cohort, provider))
    if seen != {(cohort, provider) for cohort in cohorts for provider in BMRS}:
        msg = "focused completion manifest task set is incomplete"
        raise ValueError(msg)
    return sorted(cohorts)


def load_postprocess(inputs: FocusedInputs) -> dict[str, Any]:
    """Verify the sealed postprocess manifest and return it."""
    manifest = _read_json(inputs.postprocess_root / "postprocess_manifest.json")
    if manifest.get("contract") != POSTPROCESS_CONTRACT:
        msg = "postprocess manifest has an unexpected contract"
        raise ValueError(msg)
    return manifest


def _inference_frame(
    inputs: FocusedInputs,
    cohort: str,
    postprocess_manifest: dict[str, Any],
) -> pd.DataFrame:
    """Read one cohort's sealed inference after verifying its full hash chain."""
    records = {
        str(record.get("path")): record
        for record in postprocess_manifest.get("cohort_manifests", [])
    }
    manifest_record = records.get(f"{cohort}/cohort_manifest.json", {})
    manifest_path = inputs.postprocess_root / cohort / "cohort_manifest.json"
    if manifest_record.get("sha256") != sha256_file(manifest_path):
        msg = f"postprocess cohort manifest changed: {cohort}"
        raise ValueError(msg)
    cohort_manifest = _read_json(manifest_path)
    path = inputs.postprocess_root / cohort / postprocess.RESULT_NAME
    if cohort_manifest.get("cohort") != cohort or cohort_manifest.get("output", {}).get(
        "sha256",
    ) != sha256_file(path):
        msg = f"provider inference changed since postprocess: {cohort}"
        raise ValueError(msg)
    for provider in BMRS:
        source = cohort_manifest.get("sources", {}).get(provider, {})
        task_path = inputs.run_root / str(source.get("path"))
        if (
            source.get("path")
            != f"tasks/{cohort}/{provider}/pairwise_interaction_results.csv"
            or source.get("sha256") != sha256_file(task_path)
        ):
            msg = f"postprocess was not computed from this task: {cohort}/{provider}"
            raise ValueError(msg)
    frame = pd.read_csv(path, float_precision="round_trip")
    postprocess.validate_inference_frame(frame, cohort=cohort)
    return frame


def _task_frame(inputs: FocusedInputs, cohort: str, provider: str) -> pd.DataFrame:
    path = (
        inputs.run_root
        / "tasks"
        / cohort
        / provider
        / "pairwise_interaction_results.csv"
    )
    return pd.read_csv(path, float_precision="round_trip")


def _load_counts(path: Path, *, label: str) -> pd.DataFrame:
    counts = pd.read_csv(path, index_col=0)
    values = counts.to_numpy(dtype=np.float64)
    if (
        counts.empty
        or not counts.index.is_unique
        or not counts.columns.is_unique
        or not np.isfinite(values).all()
        or (values < 0).any()
        or not np.equal(values, np.floor(values)).all()
    ):
        msg = f"{label}: invalid count matrix"
        raise ValueError(msg)
    return counts.astype(np.int64)


def _cbio_url(cohort: str) -> str:
    study_id = "coadread" if cohort == "CRAD" else cohort.lower()
    return (
        "https://www.cbioportal.org/study/summary?id="
        f"{study_id}_tcga_pan_can_atlas_2018"
    )


def _canonical_pairs(features: list[str]) -> list[tuple[int, int]]:
    return [
        (a, b)
        for a, b in combinations(range(len(features)), 2)
        if features[a].rsplit("_", 1)[0] != features[b].rsplit("_", 1)[0]
    ]


def _table_s5(inputs: FocusedInputs) -> dict[str, dict[str, str]] | None:
    if inputs.report_root is None:
        return None
    manifest = _read_json(inputs.report_root / "report_manifest.json")
    if manifest.get("inference_status") != "reportable":
        msg = "revision report is not reportable"
        raise ValueError(msg)
    table = pd.read_csv(inputs.report_root / "table_s5.csv", dtype=str)
    return {row["cohort"]: row for row in table.to_dict("records")}


def _table_s5_expectations(row: dict[str, str]) -> dict[str, int]:
    expected = {
        "tumors": int(row["tumors"]),
        "tested_pairs": int(row["tested_pairs"]),
        "same_base_pairs_excluded": int(row["same_base_pairs_excluded"]),
    }
    for provider in BMRS:
        for adjustment, analysis in (("by", "primary"), ("bh", "sensitivity")):
            if provider == "mutsig" and analysis == "primary":
                prefix = "mutsig_primary_rejection"
            else:
                prefix = f"{provider}_descriptive_{analysis}_rule_crossing"
            for suffix in ("total", "me", "co", "direction_unavailable"):
                expected[f"{provider}:{adjustment}:{suffix}"] = int(
                    row[f"{prefix}_{suffix}"],
                )
    return expected


# --------------------------------------------------------------------------- #
#                                cohort assembly                              #
# --------------------------------------------------------------------------- #
def _model_columns(
    *,
    provider: str,
    inference: pd.DataFrame,
    task: pd.DataFrame,
    identifiability_codes: dict[str, int],
    label: str,
) -> dict[str, np.ndarray]:
    for source, target in (
        ("Likelihood Ratio", "likelihood_ratio"),
        ("Rho", "rho"),
        ("Fit Iterations", "fit_iterations"),
        ("Fit Converged", "fit_converged"),
    ):
        if not np.array_equal(
            task[source].to_numpy(),
            inference[f"{provider}_{target}"].to_numpy(),
            equal_nan=True,
        ):
            msg = f"{label}: task {source} differs from sealed inference"
            raise ValueError(msg)
    identifiability = inference[f"{provider}_effect_identifiability"].astype(str)
    if not np.array_equal(
        task["Effect Identifiability"].astype(str).to_numpy(),
        identifiability.to_numpy(),
    ):
        msg = f"{label}: task identifiability differs from sealed inference"
        raise ValueError(msg)
    labels = inference[f"{provider}_direction"].astype(str).to_numpy()
    unknown = set(labels) - set(DIRECTION_CODES)
    if unknown:
        msg = f"{label}: unknown direction labels {sorted(unknown)}"
        raise ValueError(msg)

    columns: dict[str, np.ndarray] = {}
    for name, dtype, source in MODEL_COLUMNS:
        if source == "derived":
            continue
        kind, key = source.split(":", 1)
        series = inference[f"{provider}_{key}"] if kind == "inference" else task[key]
        if dtype == "uint8":
            if series.dtype != bool:
                msg = f"{label}: {key} must be boolean"
                raise ValueError(msg)
            values = series
        else:
            values = pd.to_numeric(series, errors="raise")
        columns[name] = values.to_numpy(dtype=NUMPY_DTYPES[dtype].type)
    required_finite = ("lrt", "log_p", "p", "log_by_q", "by_q", "log_bh_q", "bh_q")
    for name in (*required_finite, "tau00", "tau10", "tau01", "tau11"):
        if not np.isfinite(columns[name]).all():
            msg = f"{label}: non-finite {name}"
            raise ValueError(msg)
    direction = np.array([DIRECTION_CODES[value] for value in labels], dtype=np.uint8)
    rho = columns["rho"]
    finite_rho = np.isfinite(rho)
    expected_direction = np.full(
        len(rho),
        DIRECTION_CODES["unavailable"],
        dtype=np.uint8,
    )
    expected_direction[finite_rho & (rho < 0)] = DIRECTION_CODES["ME"]
    expected_direction[finite_rho & (rho > 0)] = DIRECTION_CODES["CO"]
    expected_direction[finite_rho & (rho == 0)] = DIRECTION_CODES["neutral"]
    if not np.array_equal(direction, expected_direction):
        msg = f"{label}: direction labels are not the rho sign"
        raise ValueError(msg)
    columns["direction"] = direction
    columns["identifiability"] = identifiability.map(identifiability_codes).to_numpy(
        dtype=np.uint8,
    )
    columns["by_decisions"] = decision_bits(columns["log_by_q"])
    columns["bh_decisions"] = decision_bits(columns["log_bh_q"])
    columns["rank"] = direction_ranks(
        direction,
        columns["log_by_q"],
        columns["log_p"],
        np.nan_to_num(rho, nan=0.0),
    )
    return columns


def _model_summary(
    columns: dict[str, np.ndarray],
    full_rank_code: int,
) -> dict[str, Any]:
    direction = columns["direction"]
    primary_bit = 1 << DECISION_THRESHOLDS.index(0.01)
    summary: dict[str, Any] = {
        "tested_pairs": len(direction),
        "identifiable_pairs": int(np.sum(columns["identifiability"] == full_rank_code)),
        "directions": {
            label: int(np.sum(direction == code))
            for label, code in DIRECTION_CODES.items()
        },
    }
    for adjustment in ADJUSTMENTS:
        crossing = (columns[f"{adjustment}_decisions"] & primary_bit) > 0
        summary[f"{adjustment}_q_le_0_01"] = {
            "total": int(crossing.sum()),
            "ME": int(np.sum(crossing & (direction == DIRECTION_CODES["ME"]))),
            "CO": int(np.sum(crossing & (direction == DIRECTION_CODES["CO"]))),
            "direction_unavailable": int(
                np.sum(
                    crossing
                    & (direction != DIRECTION_CODES["ME"])
                    & (direction != DIRECTION_CODES["CO"]),
                ),
            ),
        }
    return summary


def _baseline_columns(  # noqa: PLR0913
    *,
    baseline_root: Path,
    baseline_entry: dict[str, Any],
    cohort_id: str,
    contract: dict[str, Any],
    count_sha256: str,
    pair_keys: list[tuple[str, str]],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    cohort = cohort_id.split("__", 1)[1]
    directory = baseline_root / STUDY / cohort
    comparison = directory / "comparison_pairwise_interaction_results.csv"
    metadata_path = directory / "metadata.json"
    metadata = _read_json(metadata_path)
    artifacts = baseline_entry.get("artifacts", {})
    if (
        baseline_entry.get("id") != cohort_id
        or metadata.get("cohort", {}).get("id") != cohort_id
        or metadata.get("source_gene_k") != TOP_K
        or metadata.get("input", {}).get("sha256") != count_sha256
        or metadata.get("feature_axis", {}).get("ordered_features_sha256")
        != contract.get("ordered_features_sha256")
        or metadata.get("feature_axis", {}).get("exclude_same_base_pairs") is not True
        or artifacts.get("comparison", {}).get("sha256") != sha256_file(comparison)
        or artifacts.get("metadata", {}).get("sha256") != sha256_file(metadata_path)
    ):
        msg = f"{cohort_id}: baseline results are not bound to the focused contract"
        raise ValueError(msg)
    frame = pd.read_csv(comparison, float_precision="round_trip")
    keys = [
        tuple(sorted((str(a), str(b))))
        for a, b in zip(frame["Gene A"], frame["Gene B"], strict=True)
    ]
    position = {key: index for index, key in enumerate(keys)}
    if len(position) != len(keys) or set(position) != {
        tuple(sorted(key)) for key in pair_keys
    }:
        msg = f"{cohort_id}: baseline pair family differs from the DIALECT family"
        raise ValueError(msg)
    order = np.array(
        [position[tuple(sorted(key))] for key in pair_keys],
        dtype=np.int64,
    )
    columns = {}
    for name, source in BASELINE_COLUMNS:
        values = frame[source].to_numpy(dtype=np.float64)[order]
        if not np.isfinite(values).all():
            msg = f"{cohort_id}: non-finite baseline {source}"
            raise ValueError(msg)
        if name != "megsa_lrt" and ((values < 0) | (values > 1)).any():
            msg = f"{cohort_id}: baseline {source} outside [0, 1]"
            raise ValueError(msg)
        columns[name] = values
    fdr = 0.01
    summary = {
        "tested_pairs": len(order),
        "significant_calls": {
            "fisher_me": int((columns["fisher_me_q"] < fdr).sum()),
            "fisher_co": int((columns["fisher_co_q"] < fdr).sum()),
            "discover_me": int((columns["discover_me_q"] < fdr).sum()),
            "discover_co": int((columns["discover_co_q"] < fdr).sum()),
            "megsa_me": int((columns["megsa_p"] < 0.001).sum()),
            "wesme_me": int((columns["wesme_q"] < fdr).sum()),
            "wesco_co": int((columns["wesco_q"] < fdr).sum()),
        },
    }
    return columns, {
        "summary": summary,
        "comparison": {
            "path": comparison.as_posix(),
            "sha256": sha256_file(comparison),
        },
        "metadata": {
            "path": metadata_path.as_posix(),
            "sha256": sha256_file(metadata_path),
        },
        "seed": metadata.get("rng", {}).get("seed"),
    }


def _cohort(  # noqa: PLR0913
    *,
    inputs: FocusedInputs,
    cohort: str,
    postprocess_manifest: dict[str, Any],
    directory: Path,
    baseline_entry: dict[str, Any],
    drivers: set[str],
    table_s5: dict[str, dict[str, str]] | None,
) -> dict[str, Any]:
    cohort_id = f"{STUDY}__{cohort}"
    contract_path = inputs.run_root / "contracts" / f"{cohort}.json"
    contract = _read_json(contract_path)
    features = [str(feature) for feature in contract.get("features", [])]
    if contract.get("top_k") != TOP_K or len(features) != TOP_K:
        msg = f"{cohort_id}: contract is not an exact K={TOP_K} axis"
        raise ValueError(msg)
    count_path = inputs.provider_root / "cohorts" / cohort / "count_matrix.csv"
    count_sha256 = sha256_file(count_path)
    if contract.get("inputs", {}).get("counts", {}).get("sha256") != count_sha256:
        msg = f"{cohort_id}: count matrix is not the contract input"
        raise ValueError(msg)
    counts = _load_counts(count_path, label=cohort_id)
    n_samples = len(counts)
    if n_samples > MAX_UINT16:
        msg = f"{cohort_id}: {n_samples} samples exceed the uint16 contingency encoding"
        raise ValueError(msg)

    pairs = _canonical_pairs(features)
    if len(pairs) != contract.get("pair_policy", {}).get("row_count"):
        msg = f"{cohort_id}: canonical pair family differs from the contract"
        raise ValueError(msg)
    pair_a = np.array([a for a, _ in pairs], dtype=np.int64)
    pair_b = np.array([b for _, b in pairs], dtype=np.int64)
    pair_keys = [(features[a], features[b]) for a, b in pairs]

    inference = _inference_frame(inputs, cohort, postprocess_manifest)
    if list(zip(inference["gene_a"], inference["gene_b"], strict=True)) != pair_keys:
        msg = f"{cohort_id}: sealed inference rows are not in canonical pair order"
        raise ValueError(msg)

    binary = (counts[features].to_numpy() > 0).astype(np.int64)
    cooccurrence = binary.T @ binary
    marginals = binary.sum(axis=0)
    both = cooccurrence[pair_a, pair_b]
    pair_columns = {
        "a": pair_a,
        "b": pair_b,
        "observed_both": both,
        "observed_a_only": marginals[pair_a] - both,
        "observed_b_only": marginals[pair_b] - both,
        "observed_neither": n_samples - marginals[pair_a] - marginals[pair_b] + both,
    }
    observed = np.column_stack(
        [
            pair_columns[name]
            for name in (
                "observed_both",
                "observed_a_only",
                "observed_b_only",
                "observed_neither",
            )
        ],
    )

    identifiability_labels: set[str] = set()
    tasks = {}
    for provider in BMRS:
        task = _task_frame(inputs, cohort, provider)
        if list(zip(task["Gene A"], task["Gene B"], strict=True)) != pair_keys:
            msg = f"{cohort_id}/{provider}: task rows are not in canonical pair order"
            raise ValueError(msg)
        if not np.array_equal(
            task[["_11_", "_10_", "_01_", "_00_"]].to_numpy(),
            observed,
        ):
            msg = f"{cohort_id}/{provider}: contingency counts disagree with counts"
            raise ValueError(msg)
        identifiability_labels.update(
            inference[f"{provider}_effect_identifiability"].astype(str),
        )
        tasks[provider] = task
    identifiability = sorted(identifiability_labels)
    identifiability_codes = {label: code for code, label in enumerate(identifiability)}
    full_rank_code = identifiability_codes.get("full-affine-rank", -1)

    directory.mkdir(parents=True, exist_ok=False)
    pair_spec = PAIR_COLUMNS
    tables = {"pairs": _write_table(directory, "pairs", pair_columns, pair_spec)}
    model_spec = tuple((name, dtype) for name, dtype, _ in MODEL_COLUMNS)
    summaries = {}
    for provider in BMRS:
        columns = _model_columns(
            provider=provider,
            inference=inference,
            task=tasks[provider],
            identifiability_codes=identifiability_codes,
            label=f"{cohort_id}/{provider}",
        )
        summaries[provider] = _model_summary(columns, full_rank_code)
        tables[provider] = _write_table(
            directory,
            f"dialect-{provider}",
            columns,
            model_spec,
        )

    baseline_columns, baseline_record = _baseline_columns(
        baseline_root=inputs.baseline_root,
        baseline_entry=baseline_entry,
        cohort_id=cohort_id,
        contract=contract,
        count_sha256=count_sha256,
        pair_keys=pair_keys,
    )
    tables["baselines"] = _write_table(
        directory,
        "baselines",
        baseline_columns,
        tuple((name, "float64") for name, _ in BASELINE_COLUMNS),
    )

    same_base = TOP_K * (TOP_K - 1) // 2 - len(pairs)
    if table_s5 is not None:
        expected = _table_s5_expectations(table_s5[cohort])
        observed_facts = {
            "tumors": n_samples,
            "tested_pairs": len(pairs),
            "same_base_pairs_excluded": same_base,
        }
        for provider in BMRS:
            for adjustment in ADJUSTMENTS:
                crossing = summaries[provider][f"{adjustment}_q_le_0_01"]
                observed_facts[f"{provider}:{adjustment}:total"] = crossing["total"]
                observed_facts[f"{provider}:{adjustment}:me"] = crossing["ME"]
                observed_facts[f"{provider}:{adjustment}:co"] = crossing["CO"]
                observed_facts[f"{provider}:{adjustment}:direction_unavailable"] = (
                    crossing["direction_unavailable"]
                )
        mismatched = {
            key: (observed_facts[key], value)
            for key, value in expected.items()
            if observed_facts[key] != value
        }
        if mismatched:
            msg = (
                f"{cohort_id}: release disagrees with published Table S5: {mismatched}"
            )
            raise ValueError(msg)

    observed_genes = {feature.rsplit("_", 1)[0] for feature in features}
    payload = {
        "id": cohort_id,
        "study": STUDY,
        "cohort": cohort,
        "k": TOP_K,
        "n_samples": n_samples,
        "features": features,
        "drivers": sorted(observed_genes & drivers),
        "pair_policy": {
            "construction": "all unordered pairs of the ordered feature axis, a < b",
            "same_base_missense_nonsense": "excluded before fitting and testing",
            "same_base_pairs_excluded": same_base,
            "tested_pairs": len(pairs),
        },
        "enums": {"direction": list(DIRECTIONS), "identifiability": identifiability},
        "decision_thresholds": list(DECISION_THRESHOLDS),
        "tables": tables,
        "summaries": {"models": summaries, "baselines": baseline_record["summary"]},
        "provenance": {
            "contract": _file_record(contract_path),
            "count_matrix": _file_record(count_path),
            "inference": _file_record(
                inputs.postprocess_root / cohort / postprocess.RESULT_NAME,
            ),
            "tasks": {
                provider: _file_record(
                    inputs.run_root
                    / "tasks"
                    / cohort
                    / provider
                    / "pairwise_interaction_results.csv",
                )
                for provider in BMRS
            },
            "baselines": {
                key: baseline_record[key] for key in ("comparison", "metadata", "seed")
            },
        },
    }
    cohort_path = directory / "cohort.json"
    _write_json(cohort_path, payload)
    return {
        "id": cohort_id,
        "study": STUDY,
        "cohort": cohort,
        "cancer": TCGA_CANCER_NAMES.get(cohort, cohort),
        "k": TOP_K,
        "n_samples": n_samples,
        "median_mutations": float(counts.sum(axis=1).median()),
        "cbio": _cbio_url(cohort),
        "data_file": f"cohorts/{cohort_id}/cohort.json",
        "data_sha256": sha256_file(cohort_path),
        "data_bytes": cohort_path.stat().st_size,
        "table_bytes": sum(table["bytes"] for table in tables.values()),
        "model_summaries": summaries,
        "baseline_summary": baseline_record["summary"],
    }


# --------------------------------------------------------------------------- #
#                                 release level                               #
# --------------------------------------------------------------------------- #
def _write_readme(path: Path, release_id: str, cohorts: int) -> None:
    path.write_text(
        f"""# DIALECT Atlas {release_id}

Complete K=500 release for {cohorts} TCGA PanCancer Atlas cohorts, matching the
tested family of the DIALECT manuscript revision. MSK-IMPACT and MSK-CHORD cohorts
were not fitted at K=500 and remain in the K=100 release (`k100-2026-08-26`).

## Files

- `manifest.json`: analysis contract (the frozen revision reporting rule), encoding,
  coverage, and provenance with exact source hashes.
- `index.json`: cohort metadata, per-model and baseline summaries, and hashes.
- `cohorts/TCGA__<cohort>/cohort.json`: the ordered 500-feature axis, drivers, enums,
  and a record (rows, column layout, byte counts, SHA-256 of compressed and raw bytes)
  for each binary table.
- `cohorts/TCGA__<cohort>/pairs.bin.gz`: every tested pair as feature-axis indices
  (`a < b`) plus observed contingency counts.
- `cohorts/TCGA__<cohort>/dialect-{{mutsig,cbase,dig}}.bin.gz`: every fitted pair
  for one background model, row-aligned to `pairs`.
- `cohorts/TCGA__<cohort>/baselines.bin.gz`: Fisher, DISCOVER, MEGSA, and WeSME/WeSCO
  on the same pair family, row-aligned to `pairs`.

## Encoding

Tables are gzip-compressed. Decompressed, each column is stored contiguously and
little-endian at the byte offset listed in `cohort.json`; float64 columns precede
uint32, uint16, and uint8 columns, so every offset is aligned to its item size. NaN
encodes null (for example `rho` where the effect is not identifiable). `direction`
and `identifiability` are codes into `cohort.json` `enums`. `by_decisions` and
`bh_decisions` are bit masks: bit i is set when `log q <= log t` for the i-th value
of `decision_thresholds` (0.001, 0.005, 0.01, 0.05).

## DIALECT inference (frozen revision rule)

Each cohort tests every unordered pair of its frozen 500-feature axis, excluding
same-gene missense/nonsense pairs. For each background model separately, `p` is the
chi-square(1) profile-LRT probability when the pair effect is identifiable
(`full-affine-rank`) and 1 otherwise. Benjamini-Yekutieli over the complete
within-cohort family is primary; Benjamini-Hochberg is a nominal sensitivity.
Decisions use natural-log q-values, inclusively. MutSigCV2 is the only inferential
background: an `MutSigCV2 BY q <= 0.01` crossing is a rejection. CBaSE and DIG
crossings are descriptive continuity and sensitivity comparisons, and agreement
across backgrounds is descriptive, not an inferential vote. Direction is the rho sign
after a rejection; rejections without a defined sign are kept in the family and
excluded from ME/CO lists. This rule was calibrated before any association output
was read (two-stage fitted-null confirmation, total familywise error 0.05); the
calibration is a finite-scenario stress test, not a formal uniform FDR proof.

`rank` orders each background's ME and CO rows by BY log q, then log p, then |rho|
(descending), then pair order. Values are the sealed revision postprocess outputs
verbatim, and per-cohort crossing counts equal the published Table S5.

## Comparison methods

Fisher, DISCOVER, WeSME, and WeSCO use direction-specific BH `q < 0.01`; MEGSA is
ME-only with `p < 0.001`. DISCOVER q-values are recomputed with BH over exactly this
pair family. See `manifest.json` for seeds and versions.
""",
        encoding="utf-8",
    )


def _baseline_manifest(inputs: FocusedInputs) -> tuple[dict[str, Any], Path]:
    path = inputs.baseline_root / "manifest.json"
    manifest = _read_json(path)
    profile = manifest.get("profile", {})
    if (
        manifest.get("release_id") != BASELINE_RELEASE_ID
        or manifest.get("source_gene_k") != TOP_K
        or manifest.get("cohort_count") != inputs.expected_cohorts
        or profile.get("exclude_same_base_pairs") is not True
    ):
        msg = f"baseline release is not the focused K={TOP_K} profile: {path}"
        raise ValueError(msg)
    return manifest, path


def _baseline_sources(manifest: dict[str, Any], *, require_committed: bool) -> None:
    hashes = manifest.get("provenance", {}).get("source_files", {})
    if not hashes:
        msg = "baseline release does not publish its implementation source hashes"
        raise ValueError(msg)
    paths = []
    for path_string, expected in hashes.items():
        path = Path(path_string)
        if path.is_absolute() or ".." in path.parts:
            msg = f"invalid baseline source path: {path_string}"
            raise ValueError(msg)
        if not path.is_file() or sha256_file(path) != expected:
            msg = f"baseline source snapshot mismatch: {path_string}"
            raise ValueError(msg)
        paths.append(path)
    if require_committed:
        _require_committed_sources(tuple(paths))


def _build_tree(
    *,
    out: Path,
    inputs: FocusedInputs,
    release_id: str,
    generated_at: str | None,
    require_committed_sources: bool,
) -> dict[str, Any]:
    if require_committed_sources:
        _require_committed_sources(RELEASE_SOURCE_FILES)
    rule = load_rule(inputs)
    cohorts = load_completion(inputs)
    postprocess_manifest = load_postprocess(inputs)
    if sorted(postprocess_manifest.get("cohorts", [])) != cohorts:
        msg = "postprocess cohorts differ from the completed grid"
        raise ValueError(msg)
    baseline_manifest, baseline_manifest_path = _baseline_manifest(inputs)
    _baseline_sources(baseline_manifest, require_committed=require_committed_sources)
    baseline_entries = {
        str(entry.get("id")): entry for entry in baseline_manifest.get("cohorts", [])
    }
    if set(baseline_entries) != {f"{STUDY}__{cohort}" for cohort in cohorts}:
        msg = "baseline release does not exactly cover the focused cohorts"
        raise ValueError(msg)
    drivers_table = pd.read_csv(inputs.drivers_path, sep="\t")
    drivers = set(drivers_table["Hugo Symbol"].astype(str))
    table_s5 = _table_s5(inputs)
    if table_s5 is not None and set(table_s5) != set(cohorts):
        msg = "published Table S5 does not cover the focused cohorts"
        raise ValueError(msg)

    records = [
        _cohort(
            inputs=inputs,
            cohort=cohort,
            postprocess_manifest=postprocess_manifest,
            directory=out / "cohorts" / f"{STUDY}__{cohort}",
            baseline_entry=baseline_entries[f"{STUDY}__{cohort}"],
            drivers=drivers,
            table_s5=table_s5,
        )
        for cohort in cohorts
    ]

    index_path = out / "index.json"
    _write_json(index_path, {"release_id": release_id, "cohorts": records})
    readme_path = out / "README.md"
    _write_readme(readme_path, release_id, len(records))
    if require_committed_sources:
        _require_committed_sources(RELEASE_SOURCE_FILES)
    receipt = (
        _read_json(inputs.release_receipt)
        if inputs.release_receipt is not None and inputs.release_receipt.is_file()
        else None
    )
    rejections = Counter()
    for record in records:
        crossing = record["model_summaries"]["mutsig"]["by_q_le_0_01"]
        rejections.update(
            {key: crossing[key] for key in ("ME", "CO", "direction_unavailable")},
        )
    manifest = {
        "release_id": release_id,
        "schema_version": SCHEMA_VERSION,
        "immutable": True,
        "generated_at": generated_at or _utc_now(),
        "title": "DIALECT Atlas complete K=500 revision release",
        "coverage": {
            "cohorts": len(records),
            "studies": {STUDY: len(records)},
            "samples": sum(record["n_samples"] for record in records),
            "tested_pairs_per_model": sum(
                record["model_summaries"]["mutsig"]["tested_pairs"]
                for record in records
            ),
            "dialect_tables": len(records) * len(BMRS),
            "baseline_tables": len(records),
            "excluded_studies": {
                "MSK-IMPACT": "not fitted at K=500; see k100-2026-08-26",
                "MSK-CHORD": "not fitted at K=500; see k100-2026-08-26",
            },
            "mutsig_primary_rejections": dict(rejections),
        },
        "analysis": {
            "top_k_event_features": TOP_K,
            "fdr_threshold": rule["primary_q_threshold"],
            "fdr_operator": "<=",
            "primary_provider": rule["primary_provider"],
            "primary_adjustment": rule["primary_adjustment"],
            "sensitivity_adjustment": rule["sensitivity_adjustment"],
            "primary_q_threshold": rule["primary_q_threshold"],
            "sensitivity_q_threshold": rule["sensitivity_q_threshold"],
            "decision_scale": "natural-log q-values, inclusive",
            "decision_thresholds": list(DECISION_THRESHOLDS),
            "test": rule["test"],
            "effective_p_policy": rule["effective_p_policy"],
            "multiplicity": rule["multiplicity"],
            "direction": rule["direction"],
            "direction_unavailable": rule["direction_unavailable"],
            "provider_overlap": rule["provider_overlap"],
            "claim_scope": rule["claim_scope"],
            "calibration": {
                "interpretation": rule["calibration_interpretation"],
                "gate": rule["calibration_gate"],
            },
            "ranking": (
                "per background and direction: BY log q ascending, log p ascending, "
                "|rho| descending, canonical pair order"
            ),
            "pair_family": (
                "every unordered pair of the frozen 500-feature axis; same-gene "
                "missense/nonsense pairs excluded before fitting"
            ),
        },
        "encoding": {
            "format": "columnar-little-endian-gzip-v1",
            "compression": "gzip (RFC 1952), level 9, mtime 0",
            "column_order": list(DTYPE_ORDER),
            "null": "IEEE-754 NaN",
            "pair_index": "row i of every table is pair i of pairs.bin.gz",
        },
        "bmrs": [{"id": bmr, **BMR_METADATA[bmr]} for bmr in BMRS],
        "methods": {
            "dialect": {
                "directions": ["ME", "CO"],
                "multiple_testing_family": "one nondirectional family per background",
            },
            "fisher": {"directions": ["ME", "CO"], "call_rule": "BH q < 0.01"},
            "discover": {
                "directions": ["ME", "CO"],
                "call_rule": "BH q < 0.01",
                "version": baseline_manifest.get("provenance", {})
                .get("discover", {})
                .get("version"),
                "python_source_sha256": baseline_manifest.get("provenance", {})
                .get("discover", {})
                .get("python_source_sha256"),
            },
            "megsa": {"directions": ["ME"], "call_rule": "p < 0.001"},
            "wesme_wesco": {
                "directions": ["ME", "CO"],
                "call_rule": "BH q < 0.01",
                "seeded": True,
            },
        },
        "provenance": {
            "dialect_repository": "https://github.com/raphael-group/dialect",
            "release_assembly_commit": _git_revision(),
            "source_snapshot": {
                path.as_posix(): sha256_file(path) for path in RELEASE_SOURCE_FILES
            },
            "reporting_rule": _file_record(inputs.rule_path),
            "calibration_confirmation_summary": _file_record(
                inputs.confirmation_summary,
            ),
            "completion_manifest": _file_record(
                inputs.run_root / "completion_manifest.json",
            ),
            "run_manifest": _file_record(inputs.run_root / "run_manifest.json"),
            "postprocess_manifest": _file_record(
                inputs.postprocess_root / "postprocess_manifest.json",
            ),
            "report_manifest": (
                _file_record(inputs.report_root / "report_manifest.json")
                if inputs.report_root is not None
                else None
            ),
            "table_s5": (
                _file_record(inputs.report_root / "table_s5.csv")
                if inputs.report_root is not None
                else None
            ),
            "focused_release_receipt": receipt,
            "driver_reference": _file_record(inputs.drivers_path),
            "baseline_release": {
                **_file_record(baseline_manifest_path),
                "release_id": baseline_manifest.get("release_id"),
                "release_seed": baseline_manifest.get("release_seed"),
                "git": baseline_manifest.get("provenance", {}).get("git"),
                "source_files": baseline_manifest.get("provenance", {}).get(
                    "source_files",
                ),
            },
        },
        "index_file": "index.json",
        "index_sha256": sha256_file(index_path),
        "index_bytes": index_path.stat().st_size,
        "readme_file": "README.md",
        "readme_sha256": sha256_file(readme_path),
        "readme_bytes": readme_path.stat().st_size,
    }
    _write_json(out / "manifest.json", manifest)
    return manifest


def build_release(
    *,
    out: Path,
    release_id: str,
    inputs: FocusedInputs | None = None,
    generated_at: str | None = None,
    require_committed_sources: bool = True,
) -> dict[str, Any]:
    """Build the K=500 release atomically, refusing every overwrite."""
    if out.exists():
        msg = f"immutable release target already exists: {out}"
        raise FileExistsError(msg)
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=out.parent))
    try:
        manifest = _build_tree(
            out=staging,
            inputs=inputs or FocusedInputs(),
            release_id=release_id,
            generated_at=generated_at,
            require_committed_sources=require_committed_sources,
        )
        verify_release(staging)
        staging.rename(out)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def verify_release(root: Path) -> dict[str, int]:
    """Re-read a K=500 release and check every hash, layout, and row invariant."""
    manifest = _read_json(root / "manifest.json")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("immutable") is not True
    ):
        msg = "not an immutable K=500 release manifest"
        raise ValueError(msg)
    for kind in ("index", "readme"):
        path = root / manifest[f"{kind}_file"]
        if sha256_file(path) != manifest[f"{kind}_sha256"]:
            msg = f"{kind} hash mismatch"
            raise ValueError(msg)
    index = _read_json(root / manifest["index_file"])
    if index.get("release_id") != manifest["release_id"]:
        msg = "index/manifest release mismatch"
        raise ValueError(msg)
    tables_checked = 0
    for record in index["cohorts"]:
        cohort_path = root / record["data_file"]
        if sha256_file(cohort_path) != record["data_sha256"]:
            msg = f"{record['id']}: cohort.json hash mismatch"
            raise ValueError(msg)
        cohort = _read_json(cohort_path)
        rows = cohort["pair_policy"]["tested_pairs"]
        for name, table in cohort["tables"].items():
            compressed = (cohort_path.parent / table["file"]).read_bytes()
            if sha256_bytes(compressed) != table["sha256"]:
                msg = f"{record['id']}/{name}: compressed hash mismatch"
                raise ValueError(msg)
            raw = gzip.decompress(compressed)
            if sha256_bytes(raw) != table["raw_sha256"] or table["rows"] != rows:
                msg = f"{record['id']}/{name}: raw table mismatch"
                raise ValueError(msg)
            columns = decode_table(raw, rows, table["columns"])
            if sum(len(column.tobytes()) for column in columns.values()) != len(raw):
                msg = f"{record['id']}/{name}: table has unaccounted bytes"
                raise ValueError(msg)
            if name == "pairs":
                a, b = columns["a"], columns["b"]
                if not ((a < b) & (b < len(cohort["features"]))).all():
                    msg = f"{record['id']}: invalid pair indices"
                    raise ValueError(msg)
                totals = (
                    columns["observed_both"].astype(np.int64)
                    + columns["observed_a_only"]
                    + columns["observed_b_only"]
                    + columns["observed_neither"]
                )
                if not (totals == cohort["n_samples"]).all():
                    msg = f"{record['id']}: contingency rows do not sum to n"
                    raise ValueError(msg)
            tables_checked += 1
    return {"cohorts": len(index["cohorts"]), "tables": tables_checked}
