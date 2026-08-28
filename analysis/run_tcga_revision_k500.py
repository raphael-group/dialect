"""Run the frozen, native-support TCGA K=500 revision analysis.

This runner is deliberately independent of the historical ``id_*`` output trees.
For each TCGA cohort it selects one shared, deterministic set of 500 mutation-event
features that is natively supported by CBaSE, DIG, and the per-sample MutSig lambda
tensor.  It then fits the exact same pair family under all three backgrounds while
excluding same-base-gene ``_M:_N`` pairs before fitting.

Scientific output is task-atomic and never overwritten.  A completed
``cohort/background`` task is resumed only after its manifest, hashes, ordered feature
axis, and complete ordered pair universe validate.  Failed and interrupted attempts
remain isolated under ``attempts`` / ``work`` for diagnosis.

The current repository must expose the required LRT, pair-fit, and observation-support
contracts. These intentional launch gates prevent expensive K=500 runs from using the
historical statistic, an uncertified optimizer, or silently unsupported observations.

Launch only after the LRT and signed canonical-MAF binding gates are implemented
and reviewed::

    PYTHONPATH=src /opt/anaconda3/envs/dialect/bin/python \
      -m analysis.run_tcga_revision_k500 \
      --output-root output/revision_tcga_k500_2026-08-27 --jobs 3
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import re
import resource
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy.stats import poisson

from analysis.mutsig_lambda_co import build_lambda_pmfs, load_lambda
from dialect.models import gene as gene_module
from dialect.models import interaction as interaction_module
from dialect.models.gene import Gene
from dialect.models.interaction import Interaction
from dialect.utils.identify import (
    create_single_gene_results,
    estimate_pi_for_each_gene,
    estimate_taus_for_each_interaction,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

SCHEMA_VERSION = "2.0.0"
TOP_K = 500
BMRS = ("cbase", "dig", "mutsig")
MEMORY_HEAVY_MUTSIG_COHORTS = {"BRCA", "CRAD", "LGG", "SKCM", "UCEC"}
CANARY_COHORT = "CHOL"
REQUIRED_LRT_CONTRACT = "driver-independence-constrained-mle-v1"
REQUIRED_PAIR_FIT_CONTRACT = "deterministic-simplex-coordinate-ascent-v1"
REQUIRED_PAIR_FIT_KKT_TOL = 1e-8
REQUIRED_RHO_CONTRACT = "marshall-olkin-finite-or-degenerate-null-v1"
REQUIRED_UNDEFINED_RHO_LRT_TOL = 1e-8
REQUIRED_CONTINGENCY_TABLE_CONTRACT = "observed-binary-cells-00-01-10-11-v1"
REQUIRED_LOG_ODDS_RATIO_CONTRACT = (
    "conventional-latent-odds-00x11-over-01x10-v1"
)
OBSERVATION_SUPPORT_UNIVERSE = "full-observation-support-common-universe-v1"
REQUIRED_GENE_SUPPORT_CONTRACT = "latent-state-union-v1"
SAMPLE_AXIS_CONTRACT = (
    "count-matrix-equals-authoritative-and-mutsig-patient-axis-v2"
)
MUTSIG_RECEIPT_SCHEMA_VERSION = "1"
MUTSIG_UPSTREAM_COMMIT = "0109e27e70478181695f31ca8dd281bb44f0b3af"
MUTSIG_MAF_BINDING_STATUS = "unverified-canonical-maf-stop-ship"
MUTSIG_MAF_BINDING_REQUIREMENT = (
    "bind receipt maf_sha256 to the signed canonical MAF/population manifest"
)
MUTSIG_PATCH_PATH = Path("external/mutsig2cv_octave_dialect.patch")
MUTSIG_RUNNER_PATH = Path("scripts/run_mutsig_octave.sh")
MUTSIG_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "cohort",
        "upstream_commit",
        "patch_sha256",
        "runner_sha256",
        "runtime_sha256",
        "maf_sha256",
        "sample_axis_sha256",
        "sample_axis_count",
        "lambda_sha256",
        "meta_sha256",
        "genes_sha256",
        "patients_sha256",
    },
)
TCGA_COHORTS = (
    "ACC",
    "BLCA",
    "BRCA",
    "CESC",
    "CHOL",
    "CRAD",
    "DLBC",
    "ESCA",
    "GBM",
    "HNSC",
    "KICH",
    "KIRC",
    "KIRP",
    "LAML",
    "LGG",
    "LIHC",
    "LUAD",
    "LUSC",
    "MESO",
    "OV",
    "PAAD",
    "PCPG",
    "PRAD",
    "SARC",
    "SKCM",
    "STAD",
    "TGCT",
    "THCA",
    "THYM",
    "UCEC",
    "UCS",
    "UVM",
)
FEATURE_PATTERN = re.compile(r".+_[MN]")
MUTSIG_EFFECT_INDEX = {"M": 0, "N": 1}
THREAD_LIMIT_ENV = {
    "BLIS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
INTERNAL_TASK_ENV = "DIALECT_K500_ORCHESTRATED_TASK"
PAIRWISE_COLUMNS = (
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
    "Fit Algorithm",
    "Fit Converged",
    "Fit Iterations",
    "Fit Last LL Gain",
    "Fit Fixed-Point Residual",
    "Fit KKT Residual",
    "Pair Fit Contract",
    "Null Log Likelihood",
    "Alternative Log Likelihood",
    "LRT Contract",
)
SOURCE_FILES = (
    Path("analysis/run_tcga_revision_k500.py"),
    Path("analysis/mutsig_lambda_co.py"),
    MUTSIG_PATCH_PATH,
    MUTSIG_RUNNER_PATH,
    Path("src/dialect/models/gene.py"),
    Path("src/dialect/models/interaction.py"),
    Path("src/dialect/utils/identify.py"),
)
_HASH_CHUNK_BYTES = 1024 * 1024
_GIB = 1024**3
MAX_GENERAL_JOBS = 3
PRIOR_TASK_PEAK_RSS_BYTES = round(2.083 * _GIB)
MEMORY_HEADROOM_FACTOR = 1.25
MIN_AVAILABLE_MEMORY_FRACTION = 0.33
MIN_FREE_DISK_BYTES = round(7.6 * _GIB)


@dataclass(frozen=True)
class RunPaths:
    """Input and isolated output roots for a revision run."""

    source_root: Path
    mutsig_root: Path
    output_root: Path


@dataclass(frozen=True)
class Task:
    """One atomic cohort/background inference task."""

    cohort: str
    bmr: str


@dataclass(frozen=True)
class HostResourceSnapshot:
    """One live, aggregate-only host resource readback."""

    measured_at_utc: str
    logical_cores: int
    total_memory_bytes: int
    available_memory_bytes: int
    free_disk_bytes: int
    memory_source: str


def _utc_now() -> str:
    return datetime.now(tz=UTC).isoformat()


def _task_resource_usage(started: float) -> dict[str, Any]:
    """Return normalized, explicitly sourced resource usage for this task process."""
    elapsed_seconds = time.monotonic() - started
    native_peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        native_unit = "bytes"
        peak_rss_bytes = native_peak_rss
    elif sys.platform.startswith("linux"):
        native_unit = "KiB"
        peak_rss_bytes = native_peak_rss * 1024
    else:
        msg = f"Unsupported ru_maxrss unit convention on platform {sys.platform!r}."
        raise RuntimeError(msg)
    return {
        "elapsed_seconds": elapsed_seconds,
        "peak_rss": {
            "bytes": peak_rss_bytes,
            "native_value": native_peak_rss,
            "native_unit": native_unit,
            "platform": sys.platform,
            "source": "resource.getrusage(resource.RUSAGE_SELF).ru_maxrss",
        },
    }


def _validate_task_resource_usage(manifest: dict[str, Any], task_dir: Path) -> None:
    """Fail closed when task resource provenance is absent or internally invalid."""
    usage = manifest.get("resource_usage")
    if not isinstance(usage, dict):
        msg = f"Task manifest lacks resource usage provenance: {task_dir}"
        raise TypeError(msg)
    elapsed_seconds = usage.get("elapsed_seconds")
    peak_rss = usage.get("peak_rss")
    if (
        not isinstance(elapsed_seconds, (int, float))
        or isinstance(elapsed_seconds, bool)
        or not math.isfinite(elapsed_seconds)
        or elapsed_seconds <= 0
        or not isinstance(peak_rss, dict)
    ):
        msg = f"Task manifest has invalid elapsed-time/RSS provenance: {task_dir}"
        raise ValueError(msg)
    native_value = peak_rss.get("native_value")
    peak_rss_bytes = peak_rss.get("bytes")
    platform = peak_rss.get("platform")
    native_unit = peak_rss.get("native_unit")
    if platform == "darwin":
        expected_unit = "bytes"
        multiplier = 1
    elif isinstance(platform, str) and platform.startswith("linux"):
        expected_unit = "KiB"
        multiplier = 1024
    else:
        msg = f"Task manifest has unsupported RSS platform provenance: {task_dir}"
        raise ValueError(msg)
    if (
        not isinstance(native_value, int)
        or isinstance(native_value, bool)
        or native_value <= 0
        or not isinstance(peak_rss_bytes, int)
        or isinstance(peak_rss_bytes, bool)
        or peak_rss_bytes != native_value * multiplier
        or native_unit != expected_unit
        or peak_rss.get("source")
        != "resource.getrusage(resource.RUSAGE_SELF).ru_maxrss"
    ):
        msg = f"Task manifest has invalid peak-RSS provenance: {task_dir}"
        raise ValueError(msg)


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _json_sha256(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        msg = f"Required input is missing: {path}"
        raise FileNotFoundError(msg)
    stat = path.stat()
    return {
        "path": path.as_posix(),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": _sha256(path),
    }


def _sequence_sha256(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: object, *, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("xb") as handle:
        handle.write(_canonical_json(payload))
        handle.write(b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        if replace:
            temporary.replace(path)
        else:
            os.link(temporary, path)
            temporary.unlink()
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = f"Expected a JSON object: {path}"
        raise TypeError(msg)
    return payload


def _base_gene(feature: str) -> str:
    return feature.rsplit("_", 1)[0]


def iter_tested_pairs(features: Sequence[str]) -> Iterator[tuple[str, str]]:
    """Yield the frozen ordered pair universe, omitting same-base-gene pairs."""
    for feature_a, feature_b in combinations(features, 2):
        if _base_gene(feature_a) != _base_gene(feature_b):
            yield feature_a, feature_b


def _pair_contract(features: Sequence[str]) -> dict[str, Any]:
    same_base_excluded = 0
    digest = hashlib.sha256()
    row_count = 0
    for feature_a, feature_b in combinations(features, 2):
        if _base_gene(feature_a) == _base_gene(feature_b):
            same_base_excluded += 1
            continue
        encoded = f"{feature_a}\t{feature_b}".encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        row_count += 1
    return {
        "ordered_pair_sha256": digest.hexdigest(),
        "row_count": row_count,
        "same_base_pairs_excluded": same_base_excluded,
        "unfiltered_row_count": math.comb(len(features), 2),
    }


def select_common_features(  # noqa: PLR0913
    counts: pd.DataFrame,
    *,
    cbase_features: set[str],
    dig_features: set[str],
    mutsig_genes: set[str],
    fully_supported_features: set[str] | None = None,
    top_k: int = TOP_K,
) -> tuple[list[str], dict[str, int]]:
    """Select one shared count-ranked native-support feature axis.

    Ranking is descending cohort mutation-event count. Equal totals preserve the
    original count-matrix column order, reproducing the historical CBaSE/DIG feature
    ranking while making the order immutable through the count-matrix hash.
    """
    native = {
        feature
        for feature in counts.columns
        if feature in cbase_features
        and feature in dig_features
        and FEATURE_PATTERN.fullmatch(feature) is not None
        and _base_gene(feature) in mutsig_genes
        and (
            fully_supported_features is None or feature in fully_supported_features
        )
    }
    totals = {str(feature): int(counts[feature].sum()) for feature in native}
    ordinal = {
        str(feature): position for position, feature in enumerate(counts.columns)
    }
    ordered = sorted(native, key=lambda feature: (-totals[feature], ordinal[feature]))
    if len(ordered) < top_k:
        msg = (
            f"Only {len(ordered)} features have native support in all three BMRs; "
            f"K={top_k} is impossible."
        )
        raise ValueError(msg)
    selected = ordered[:top_k]
    return selected, {feature: totals[feature] for feature in selected}


def _read_counts(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    if frame.empty or frame.shape[1] == 0:
        msg = f"Count matrix must have samples and features: {path}"
        raise ValueError(msg)
    frame.index = pd.Index([str(value) for value in frame.index])
    frame.columns = pd.Index([str(value) for value in frame.columns])
    if not frame.index.is_unique or not frame.columns.is_unique:
        msg = f"Count matrix axes must be unique: {path}"
        raise ValueError(msg)
    invalid = [
        feature
        for feature in frame.columns
        if FEATURE_PATTERN.fullmatch(feature) is None
    ]
    if invalid:
        msg = f"Count matrix contains invalid mutation-event IDs: {invalid[:5]}"
        raise ValueError(msg)
    numeric = frame.apply(pd.to_numeric, errors="raise")
    values = numeric.to_numpy(dtype=float)
    if (
        not np.isfinite(values).all()
        or (values < 0).any()
        or not np.equal(values, np.floor(values)).all()
    ):
        msg = f"Count matrix values must be finite nonnegative integers: {path}"
        raise ValueError(msg)
    return numeric.astype(np.int64)


def _load_strict_pmfs(path: Path) -> dict[str, dict[int, float]]:
    """Load exact integer-keyed PMFs without assuming contiguous count support."""
    frame = pd.read_csv(path, index_col=0)
    frame.index = pd.Index([str(value) for value in frame.index])
    if frame.empty or not frame.index.is_unique:
        msg = f"BMR PMF table must contain a unique feature axis: {path}"
        raise ValueError(msg)
    try:
        count_keys = [int(str(column)) for column in frame.columns]
    except ValueError as error:
        msg = f"BMR PMF columns must be integer count keys: {path}"
        raise ValueError(msg) from error
    if any(key < 0 for key in count_keys) or len(count_keys) != len(set(count_keys)):
        msg = f"BMR PMF count keys must be unique nonnegative integers: {path}"
        raise ValueError(msg)
    numeric = frame.apply(pd.to_numeric, errors="raise")
    values = numeric.to_numpy(dtype=float)
    finite_or_nan = np.isfinite(values) | np.isnan(values)
    if not finite_or_nan.all() or (np.nan_to_num(values, nan=0.0) < 0).any():
        msg = f"BMR PMF values must be finite, nonnegative, or padding NaN: {path}"
        raise ValueError(msg)
    row_sums = np.nansum(values, axis=1)
    if not np.isfinite(row_sums).all() or (row_sums <= 0).any():
        msg = f"Every BMR PMF row must have positive mass: {path}"
        raise ValueError(msg)
    pmfs: dict[str, dict[int, float]] = {}
    for position, feature in enumerate(frame.index):
        pmf = {
            key: float(value)
            for key, value in zip(count_keys, values[position], strict=True)
            if not np.isnan(value)
        }
        total = sum(pmf.values())
        pmfs[feature] = {key: value / total for key, value in pmf.items()}
    return pmfs


def _read_mutsig_metadata(path: Path) -> dict[str, int]:
    fields: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        pieces = line.split("\t")
        if len(pieces) != 2 or pieces[0] in fields or not pieces[1].isdigit():
            msg = f"Invalid MutSig metadata row in {path}: {line!r}"
            raise ValueError(msg)
        fields[pieces[0]] = int(pieces[1])
    if set(fields) != {"ng", "np", "neff"} or fields["neff"] != 2:
        msg = f"MutSig metadata must contain positive ng/np and neff=2: {path}"
        raise ValueError(msg)
    if fields["ng"] <= 0 or fields["np"] <= 0:
        msg = f"MutSig ng and np must be positive: {path}"
        raise ValueError(msg)
    return fields


def _read_axis(path: Path, expected: int, *, label: str) -> list[str]:
    values = path.read_text(encoding="utf-8").splitlines()
    if len(values) != expected or any(not value for value in values):
        msg = f"MutSig {label} axis does not match metadata: {path}"
        raise ValueError(msg)
    if len(values) != len(set(values)):
        msg = f"MutSig {label} axis contains duplicate identifiers: {path}"
        raise ValueError(msg)
    return values


def _read_authoritative_sample_axis(path: Path) -> list[str]:
    """Read the materialized axis in its canonical UTF-8/LF byte convention."""
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        msg = f"Authoritative sample axis is not UTF-8: {path}"
        raise ValueError(msg) from error
    values = text.splitlines()
    canonical = ("\n".join(values) + "\n").encode()
    if raw != canonical:
        msg = (
            "Authoritative sample axis must use UTF-8, LF separators, and one "
            f"terminal newline: {path}"
        )
        raise ValueError(msg)
    if not values or any(not value or value != value.strip() for value in values):
        msg = f"Authoritative sample axis contains a blank or padded ID: {path}"
        raise ValueError(msg)
    if len(values) != len(set(values)):
        msg = f"Authoritative sample axis contains duplicate IDs: {path}"
        raise ValueError(msg)
    if values != sorted(values):
        msg = f"Authoritative sample axis must be lexicographically ordered: {path}"
        raise ValueError(msg)
    return values


def _read_mutsig_receipt(path: Path) -> dict[str, str]:
    """Read the receipt published last by the tracked MutSig runner."""
    fields: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        pieces = line.split("\t")
        if (
            len(pieces) != 2
            or not pieces[0]
            or not pieces[1]
            or pieces[0] in fields
        ):
            msg = f"Invalid MutSig receipt row in {path}: {line!r}"
            raise ValueError(msg)
        fields[pieces[0]] = pieces[1]
    if set(fields) != MUTSIG_RECEIPT_KEYS:
        missing = sorted(MUTSIG_RECEIPT_KEYS - set(fields))
        unexpected = sorted(set(fields) - MUTSIG_RECEIPT_KEYS)
        msg = (
            f"MutSig receipt has the wrong fields: {path}; "
            f"missing={missing}, unexpected={unexpected}"
        )
        raise ValueError(msg)
    return fields


def _validate_mutsig_receipt(
    mutsig_dir: Path,
    dimensions: dict[str, int],
    artifact_records: dict[str, dict[str, Any]],
    *,
    authoritative_axis_sha256: str,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Validate input/source provenance and every receipt-bound sidecar."""
    receipt_path = mutsig_dir / "persample_receipt.tsv"
    receipt = _read_mutsig_receipt(receipt_path)
    if receipt["schema_version"] != MUTSIG_RECEIPT_SCHEMA_VERSION:
        msg = f"Unsupported MutSig receipt schema in {receipt_path}"
        raise ValueError(msg)
    if receipt["cohort"] != mutsig_dir.name:
        msg = f"MutSig receipt cohort does not match its directory: {receipt_path}"
        raise ValueError(msg)
    if receipt["upstream_commit"] != MUTSIG_UPSTREAM_COMMIT:
        msg = (
            "MutSig receipt does not pin the required upstream commit: "
            f"{receipt_path}"
        )
        raise ValueError(msg)
    if receipt["sample_axis_sha256"] != authoritative_axis_sha256:
        msg = (
            "MutSig receipt sample_axis_sha256 does not match the authoritative "
            f"sample_axis.txt: {receipt_path}"
        )
        raise ValueError(msg)

    sha256_fields = {key for key in MUTSIG_RECEIPT_KEYS if key.endswith("_sha256")}
    for key in sha256_fields:
        if re.fullmatch(r"[0-9a-f]{64}", receipt[key]) is None:
            msg = f"MutSig receipt {key} is not a lowercase SHA-256: {receipt_path}"
            raise ValueError(msg)
    if not receipt["sample_axis_count"].isdigit() or (
        int(receipt["sample_axis_count"]) != dimensions["np"]
    ):
        msg = (
            "MutSig receipt sample-axis count does not match metadata: "
            f"{receipt_path}"
        )
        raise ValueError(msg)

    repo_root = Path(__file__).resolve().parents[1]
    expected_source_hashes = {
        "patch_sha256": _sha256(repo_root / MUTSIG_PATCH_PATH),
        "runner_sha256": _sha256(repo_root / MUTSIG_RUNNER_PATH),
    }
    for key, expected in expected_source_hashes.items():
        if receipt[key] != expected:
            msg = (
                f"MutSig receipt {key} does not match the tracked source: "
                f"{receipt_path}"
            )
            raise ValueError(msg)

    receipt_artifact_keys = {
        "lambda": "lambda_sha256",
        "metadata": "meta_sha256",
        "genes": "genes_sha256",
        "patients": "patients_sha256",
    }
    for artifact, key in receipt_artifact_keys.items():
        if receipt[key] != artifact_records[artifact]["sha256"]:
            msg = f"MutSig receipt hash does not match {artifact}: {receipt_path}"
            raise ValueError(msg)
    return receipt, _file_record(receipt_path)


def _mutsig_contract(
    mutsig_dir: Path,
    count_samples: Sequence[str],
    *,
    authoritative_axis_sha256: str,
) -> tuple[set[str], dict[str, Any]]:
    meta_path = mutsig_dir / "persample_meta.txt"
    genes_path = mutsig_dir / "persample_genes.txt"
    patients_path = mutsig_dir / "persample_patients.txt"
    lambda_path = mutsig_dir / "persample_lambda.f32"
    fields = _read_mutsig_metadata(meta_path)
    genes = _read_axis(genes_path, fields["ng"], label="gene")
    patients = _read_axis(patients_path, fields["np"], label="patient")
    expected_bytes = fields["ng"] * fields["np"] * fields["neff"] * 4
    if lambda_path.stat().st_size != expected_bytes:
        msg = (
            f"MutSig lambda tensor has {lambda_path.stat().st_size} bytes; "
            f"metadata requires {expected_bytes}: {lambda_path}"
        )
        raise ValueError(msg)
    ordered_count_samples = [str(sample) for sample in count_samples]
    patient_position = {patient: position for position, patient in enumerate(patients)}
    count_sample_set = set(ordered_count_samples)
    patient_set = set(patients)
    missing = [sample for sample in ordered_count_samples if sample not in patient_set]
    extra = [patient for patient in patients if patient not in count_sample_set]
    if ordered_count_samples != patients:
        msg = (
            "Count-matrix sample axis must exactly equal the ordered MutSig patient "
            f"axis; missing_in_mutsig={missing[:5]}, extra_in_mutsig={extra[:5]}, "
            f"same_set={count_sample_set == patient_set}."
        )
        raise ValueError(msg)
    mapping = [
        f"{sample}\t{patient_position[sample]}" for sample in ordered_count_samples
    ]
    ordered_axis_sha256 = _sequence_sha256(ordered_count_samples)
    artifact_records = {
        "lambda": _file_record(lambda_path),
        "metadata": _file_record(meta_path),
        "genes": _file_record(genes_path),
        "patients": _file_record(patients_path),
    }
    receipt, receipt_record = _validate_mutsig_receipt(
        mutsig_dir,
        fields,
        artifact_records,
        authoritative_axis_sha256=authoritative_axis_sha256,
    )
    return set(genes), {
        "dimensions": fields,
        "receipt": {
            "schema_version": receipt["schema_version"],
            "upstream_commit": receipt["upstream_commit"],
            "patch_sha256": receipt["patch_sha256"],
            "runner_sha256": receipt["runner_sha256"],
            "runtime_sha256": receipt["runtime_sha256"],
            "maf_sha256": receipt["maf_sha256"],
            "canonical_maf_binding": {
                "status": MUTSIG_MAF_BINDING_STATUS,
                "required_before_production": MUTSIG_MAF_BINDING_REQUIREMENT,
            },
            "sample_axis_sha256": receipt["sample_axis_sha256"],
            "sample_axis_count": int(receipt["sample_axis_count"]),
        },
        "sample_mapping": {
            "contract": SAMPLE_AXIS_CONTRACT,
            "cohort_samples": len(ordered_count_samples),
            "matched_samples": len(ordered_count_samples),
            "extra_mutsig_samples": 0,
            "cohort_mean_fallback_samples": 0,
            "exact_order_match": True,
            "ordered_count_ids_sha256": ordered_axis_sha256,
            "ordered_mutsig_ids_sha256": ordered_axis_sha256,
            "ordered_mapping_sha256": _sequence_sha256(mapping),
        },
        "files": {**artifact_records, "receipt": receipt_record},
    }


def _shared_pmf_has_observation_support(
    observed: np.ndarray,
    pmf: dict[int, float],
) -> bool:
    return all(
        float(pmf.get(int(count), 0.0))
        + float(pmf.get(int(count) - 1, 0.0))
        > 0.0
        for count in np.unique(observed)
    )


def build_full_support_universe(
    counts: pd.DataFrame,
    *,
    cbase_pmfs: dict[str, dict[int, float]],
    dig_pmfs: dict[str, dict[int, float]],
    mutsig_dir: Path,
) -> tuple[set[str], dict[str, Any]]:
    """Return features with native, full observation support in all three BMRs."""
    lambdas, mutsig_genes, mutsig_patients = load_lambda(mutsig_dir)
    if not np.isfinite(lambdas).all() or (lambdas < 0).any():
        msg = f"MutSig lambda values must be finite and nonnegative: {mutsig_dir}"
        raise ValueError(msg)
    if len(mutsig_genes) != len(set(mutsig_genes)):
        msg = f"MutSig gene identifiers must be unique: {mutsig_dir}"
        raise ValueError(msg)
    if len(mutsig_patients) != len(set(mutsig_patients)):
        msg = f"MutSig patient identifiers must be unique: {mutsig_dir}"
        raise ValueError(msg)
    gene_position = {gene: position for position, gene in enumerate(mutsig_genes)}
    patient_position = {
        patient: position for position, patient in enumerate(mutsig_patients)
    }
    missing_samples = [
        str(sample) for sample in counts.index if str(sample) not in patient_position
    ]
    if missing_samples:
        msg = f"MutSig lacks count-matrix samples: {missing_samples[:5]}"
        raise ValueError(msg)
    sample_positions = np.array(
        [patient_position[str(sample)] for sample in counts.index],
        dtype=np.int64,
    )

    eligible: set[str] = set()
    exclusions: list[dict[str, Any]] = []
    provider_reason_counts = {
        provider: {"missing_native_background": 0, "zero_observation_support": 0}
        for provider in BMRS
    }
    for feature in counts.columns:
        base, effect = feature.rsplit("_", 1)
        reasons: list[dict[str, str]] = []
        if feature not in cbase_pmfs:
            reasons.append(
                {"provider": "cbase", "reason": "missing_native_background"},
            )
        if feature not in dig_pmfs:
            reasons.append(
                {"provider": "dig", "reason": "missing_native_background"},
            )
        if base not in gene_position or effect not in MUTSIG_EFFECT_INDEX:
            reasons.append(
                {"provider": "mutsig", "reason": "missing_native_background"},
            )
        if not reasons:
            observed = counts[feature].to_numpy(dtype=np.int64)
            if not _shared_pmf_has_observation_support(
                observed,
                cbase_pmfs[feature],
            ):
                reasons.append(
                    {"provider": "cbase", "reason": "zero_observation_support"},
                )
            if not _shared_pmf_has_observation_support(observed, dig_pmfs[feature]):
                reasons.append(
                    {"provider": "dig", "reason": "zero_observation_support"},
                )
            lambda_values = lambdas[
                gene_position[base],
                sample_positions,
                MUTSIG_EFFECT_INDEX[effect],
            ]
            mutsig_support = poisson.pmf(observed, lambda_values) + poisson.pmf(
                observed - 1,
                lambda_values,
            )
            if not np.isfinite(mutsig_support).all() or not (mutsig_support > 0).all():
                reasons.append(
                    {"provider": "mutsig", "reason": "zero_observation_support"},
                )
        if reasons:
            exclusions.append({"feature": feature, "reasons": reasons})
            for reason in reasons:
                provider_reason_counts[reason["provider"]][reason["reason"]] += 1
        else:
            eligible.add(feature)

    report = {
        "contract": OBSERVATION_SUPPORT_UNIVERSE,
        "support_rule": "P(B=c) + P(B=c-1) > 0 for every cohort sample",
        "count_matrix_features": len(counts.columns),
        "eligible_features": len(eligible),
        "excluded_features": exclusions,
        "excluded_feature_count": len(exclusions),
        "excluded_features_sha256": _json_sha256(exclusions),
        "provider_reason_counts": provider_reason_counts,
    }
    return eligible, report


def audit_background_support(
    counts: pd.DataFrame,
    features: Sequence[str],
    pmfs: dict[str, Any],
) -> dict[str, Any]:
    """Audit exact observed-count support and its tested-pair implications.

    A feature/sample observation is supported when at least one latent driver state
    can explain it, i.e. ``P(B=c) + P(B=c-1) > 0``. Pair support is the intersection
    of its two feature masks. The compact hashes bind every ordered feature and pair
    mask without writing a large boolean matrix to the manifest.
    """
    n_samples = len(counts)
    mask_bytes = (n_samples + 7) // 8
    sample_zero_counts = np.zeros(n_samples, dtype=np.int64)
    feature_masks: list[int] = []
    feature_mask_bytes: list[bytes] = []
    zero_by_feature: dict[str, int] = {}
    feature_digest = hashlib.sha256()

    for feature in features:
        background = pmfs.get(feature)
        if background is None:
            msg = f"Support audit is missing PMFs for {feature}."
            raise ValueError(msg)
        per_sample = (
            background if isinstance(background, list) else [background] * n_samples
        )
        if len(per_sample) != n_samples:
            msg = f"Support audit PMF/sample length mismatch for {feature}."
            raise ValueError(msg)
        observed = counts[feature].to_numpy(dtype=np.int64)
        unsupported = np.fromiter(
            (
                float(pmf.get(int(count), 0.0))
                + float(pmf.get(int(count) - 1, 0.0))
                == 0.0
                for count, pmf in zip(observed, per_sample, strict=True)
            ),
            dtype=bool,
            count=n_samples,
        )
        sample_zero_counts += unsupported
        zero_count = int(unsupported.sum())
        if zero_count:
            zero_by_feature[feature] = zero_count
        packed = np.packbits(unsupported, bitorder="little").tobytes()
        if len(packed) != mask_bytes:
            msg = "Packed support mask has an unexpected byte length."
            raise ValueError(msg)
        feature_mask_bytes.append(packed)
        feature_masks.append(int.from_bytes(packed, "little"))
        encoded = feature.encode("utf-8")
        feature_digest.update(len(encoded).to_bytes(8, "big"))
        feature_digest.update(encoded)
        feature_digest.update(packed)

    pair_digest = hashlib.sha256()
    excluded_histogram: dict[int, int] = {}
    unique_pair_masks: set[int] = set()
    pair_count = 0
    for index_a, index_b in combinations(range(len(features)), 2):
        if _base_gene(features[index_a]) == _base_gene(features[index_b]):
            continue
        mask = feature_masks[index_a] | feature_masks[index_b]
        excluded = mask.bit_count()
        excluded_histogram[excluded] = excluded_histogram.get(excluded, 0) + 1
        unique_pair_masks.add(mask)
        pair_digest.update(mask.to_bytes(mask_bytes, "little"))
        pair_count += 1

    feature_mask_count = len(set(feature_masks))
    full_pairs = excluded_histogram.get(0, 0)
    return {
        "support_rule": "P(B=c) + P(B=c-1) > 0 exactly",
        "feature_sample_observations": len(features) * n_samples,
        "zero_support_feature_samples": int(sample_zero_counts.sum()),
        "features_with_zero_support": len(zero_by_feature),
        "samples_with_any_zero_support": int(np.sum(sample_zero_counts > 0)),
        "max_zero_support_features_per_sample": int(
            sample_zero_counts.max(initial=0),
        ),
        "zero_support_by_feature": zero_by_feature,
        "ordered_feature_masks_sha256": feature_digest.hexdigest(),
        "unique_feature_masks": feature_mask_count,
        "pairs": {
            "tested": pair_count,
            "full_sample_support": full_pairs,
            "with_excluded_samples": pair_count - full_pairs,
            "excluded_sample_count_histogram": {
                str(key): excluded_histogram[key] for key in sorted(excluded_histogram)
            },
            "max_excluded_samples": max(excluded_histogram, default=0),
            "unique_effective_sample_masks": len(unique_pair_masks),
            "ordered_effective_masks_sha256": pair_digest.hexdigest(),
        },
        "inference_implication": {
            "all_features_have_full_sample_support": not any(feature_masks),
            "all_features_share_one_effective_sample_mask": feature_mask_count == 1,
            "global_feature_pi_is_exact_pair_null_on_full_sample_set": (
                not any(feature_masks)
            ),
            "pair_specific_effective_mask_refit_required": feature_mask_count > 1,
        },
    }


def _require_full_observation_support(contract: dict[str, Any]) -> None:
    universe = contract.get("full_support_universe", {})
    exclusions = universe.get("excluded_features", [])
    if universe.get("contract") != OBSERVATION_SUPPORT_UNIVERSE:
        msg = "Cohort contract does not use the frozen full-support universe."
        raise ValueError(msg)
    if universe.get("excluded_features_sha256") != _json_sha256(exclusions):
        msg = "Full-support exclusion list hash does not match the cohort contract."
        raise ValueError(msg)
    expected_pairs = contract["pair_policy"]["row_count"]
    for bmr in BMRS:
        audit = contract["observed_count_support_audit"][bmr]
        if (
            audit["zero_support_feature_samples"] != 0
            or audit["pairs"]["full_sample_support"] != expected_pairs
            or audit["pairs"]["with_excluded_samples"] != 0
            or not audit["inference_implication"][
                "global_feature_pi_is_exact_pair_null_on_full_sample_set"
            ]
        ):
            msg = (
                f"Selected {bmr} feature universe lacks full observation support; "
                "global single-gene constrained-null marginals are not valid for "
                "every tested pair."
            )
            raise ValueError(msg)


def _require_exact_sample_axis(contract: dict[str, Any]) -> None:
    """Require one authoritative ordered tumor axis across all backgrounds."""
    samples = contract.get("samples", {})
    count = samples.get("count")
    if (
        samples.get("contract") != SAMPLE_AXIS_CONTRACT
        or not samples.get("exact_order_match")
        or samples.get("extra_mutsig_samples") != 0
        or samples.get("cohort_mean_fallback_samples") != 0
        or samples.get("cohort_samples") != count
        or samples.get("authoritative_samples") != count
        or samples.get("matched_samples") != count
        or samples.get("ordered_ids_sha256")
        != samples.get("ordered_count_ids_sha256")
        or samples.get("ordered_ids_sha256")
        != samples.get("ordered_mutsig_ids_sha256")
        or samples.get("ordered_ids_sha256")
        != samples.get("ordered_authoritative_ids_sha256")
    ):
        msg = "Cohort contract does not bind one exact ordered sample axis."
        raise ValueError(msg)


def _require_canonical_mutsig_maf_binding(contract: dict[str, Any]) -> None:
    """Block production until the receipt MAF hash has an approved authority."""
    binding = (
        contract.get("inputs", {})
        .get("mutsig", {})
        .get("receipt", {})
        .get("canonical_maf_binding", {})
    )
    if binding.get("status") != "verified":
        requirement = binding.get(
            "required_before_production",
            MUTSIG_MAF_BINDING_REQUIREMENT,
        )
        msg = f"K=500 launch blocked by MutSig MAF provenance stop-ship: {requirement}."
        raise RuntimeError(msg)


def build_cohort_contract(
    paths: RunPaths,
    cohort: str,
    *,
    top_k: int = TOP_K,
) -> dict[str, Any]:
    """Build a deterministic fail-closed contract for one cohort."""
    cohort_dir = paths.source_root / cohort
    count_path = cohort_dir / "count_matrix.csv"
    sample_axis_path = cohort_dir / "sample_axis.txt"
    cbase_path = cohort_dir / "bmr_pmfs.csv"
    dig_path = cohort_dir / "bmr_pmfs.dig.csv"
    mutsig_dir = paths.mutsig_root / cohort
    counts = _read_counts(count_path)
    ordered_count_samples = [str(sample) for sample in counts.index]
    authoritative_samples = _read_authoritative_sample_axis(sample_axis_path)
    if ordered_count_samples != authoritative_samples:
        msg = (
            "Count-matrix sample axis must exactly equal authoritative "
            f"sample_axis.txt for {cohort}."
        )
        raise ValueError(msg)
    authoritative_axis_sha256 = _sha256(sample_axis_path)
    cbase_pmfs = _load_strict_pmfs(cbase_path)
    dig_pmfs = _load_strict_pmfs(dig_path)
    cbase_features = set(cbase_pmfs)
    dig_features = set(dig_pmfs)
    mutsig_genes, mutsig = _mutsig_contract(
        mutsig_dir,
        ordered_count_samples,
        authoritative_axis_sha256=authoritative_axis_sha256,
    )
    fully_supported_features, support_universe = build_full_support_universe(
        counts,
        cbase_pmfs=cbase_pmfs,
        dig_pmfs=dig_pmfs,
        mutsig_dir=mutsig_dir,
    )
    features, totals = select_common_features(
        counts,
        cbase_features=cbase_features,
        dig_features=dig_features,
        mutsig_genes=mutsig_genes,
        fully_supported_features=fully_supported_features,
        top_k=top_k,
    )
    selected_counts = counts.loc[:, features]
    selected_observed_kmax = int(selected_counts.to_numpy().max(initial=0))
    support_audit = {}
    for bmr in BMRS:
        pmfs = _task_pmfs(paths, Task(cohort, bmr), selected_counts, features)
        support_audit[bmr] = audit_background_support(
            selected_counts,
            features,
            pmfs,
        )
    contract = {
        "schema_version": SCHEMA_VERSION,
        "cohort": cohort,
        "top_k": top_k,
        "feature_policy": {
            "rank": "descending total mutation-event count",
            "tie_break": "original count-matrix column ordinal ascending",
            "support": OBSERVATION_SUPPORT_UNIVERSE,
            "mutsig_cbase_feature_fallback": False,
        },
        "pair_policy": {
            "order": "combinations of the ordered feature axis",
            "same_base_missense_nonsense": "excluded before fitting and testing",
            **_pair_contract(features),
        },
        "samples": {
            "count": len(counts),
            "authoritative_samples": len(authoritative_samples),
            "ordered_ids_sha256": _sequence_sha256(ordered_count_samples),
            "ordered_authoritative_ids_sha256": _sequence_sha256(
                authoritative_samples,
            ),
            **mutsig["sample_mapping"],
        },
        "features": features,
        "feature_totals": totals,
        "cutoff_tie": {
            "total": totals[features[-1]],
            "eligible_features": sum(
                feature in fully_supported_features
                and int(counts[feature].sum()) == totals[features[-1]]
                for feature in counts.columns
            ),
            "selected_features": sum(
                total == totals[features[-1]] for total in totals.values()
            ),
        },
        "ordered_features_sha256": _sequence_sha256(features),
        "native_support": {
            "cbase_features": len(cbase_features),
            "dig_features": len(dig_features),
            "mutsig_genes": len(mutsig_genes),
            "common_features": sum(
                feature in cbase_features
                and feature in dig_features
                and _base_gene(feature) in mutsig_genes
                for feature in counts.columns
            ),
            "full_observation_support_common_features": len(
                fully_supported_features,
            ),
            "selected_features": len(features),
        },
        "full_support_universe": support_universe,
        "mutsig_pmf_contract": {
            "native_lambda_only": True,
            "lambda_floor": None,
            "selected_observed_count_min": 0,
            "selected_observed_count_max": selected_observed_kmax,
            "poisson_count_keys": [0, selected_observed_kmax],
            "truncated_pmf_renormalized": True,
        },
        "observed_count_support_audit": support_audit,
        "inputs": {
            "counts": _file_record(count_path),
            "sample_axis": _file_record(sample_axis_path),
            "cbase": _file_record(cbase_path),
            "dig": _file_record(dig_path),
            "mutsig": mutsig,
        },
    }
    _require_exact_sample_axis(contract)
    _require_full_observation_support(contract)
    return contract


def _contract_path(paths: RunPaths, cohort: str) -> Path:
    return paths.output_root / "contracts" / f"{cohort}.json"


def _task_dir(paths: RunPaths, task: Task) -> Path:
    return paths.output_root / "tasks" / task.cohort / task.bmr


def _ensure_contract(
    paths: RunPaths,
    cohort: str,
    *,
    top_k: int = TOP_K,
) -> dict[str, Any]:
    current = build_cohort_contract(paths, cohort, top_k=top_k)
    path = _contract_path(paths, cohort)
    if path.exists():
        existing = _read_json(path)
        if existing != current:
            msg = f"Input drift detected for frozen cohort contract: {cohort}"
            raise ValueError(msg)
    else:
        _write_json_atomic(path, current)
    return current


def _verify_file_record(record: dict[str, Any]) -> None:
    path = Path(str(record["path"]))
    if not path.is_file():
        msg = f"Frozen input disappeared after preflight: {path}"
        raise FileNotFoundError(msg)
    if path.stat().st_size != int(record["bytes"]):
        msg = f"Frozen input byte length changed after preflight: {path}"
        raise ValueError(msg)
    if _sha256(path) != record["sha256"]:
        msg = f"Frozen input hash changed after preflight: {path}"
        raise ValueError(msg)


def _load_verified_contract(
    paths: RunPaths,
    cohort: str,
    *,
    top_k: int,
) -> dict[str, Any]:
    path = _contract_path(paths, cohort)
    if not path.exists():
        return _ensure_contract(paths, cohort, top_k=top_k)
    contract = _read_json(path)
    if contract.get("top_k") != top_k or contract.get("cohort") != cohort:
        msg = f"Frozen contract coordinates do not match task: {path}"
        raise ValueError(msg)
    _require_exact_sample_axis(contract)
    _require_full_observation_support(contract)
    inputs = contract["inputs"]
    _verify_file_record(inputs["counts"])
    _verify_file_record(inputs["sample_axis"])
    _verify_file_record(inputs["cbase"])
    _verify_file_record(inputs["dig"])
    for record in inputs["mutsig"]["files"].values():
        _verify_file_record(record)
    return contract


def _source_snapshot(repo_root: Path) -> dict[str, str]:
    return {
        path.as_posix(): _sha256(repo_root / path)
        for path in SOURCE_FILES
    }


def _git_snapshot(repo_root: Path) -> dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {"head": head, "dirty": bool(status), "status": status}


def _verify_recorded_source_hashes(paths: RunPaths) -> dict[str, str]:
    manifest_path = paths.output_root / "run_manifest.json"
    manifest = _read_json(manifest_path)
    recorded = manifest.get("implementation_sha256")
    if not isinstance(recorded, dict):
        msg = "Run manifest does not contain implementation source hashes."
        raise TypeError(msg)
    repo_root = Path(__file__).resolve().parents[1]
    current = _source_snapshot(repo_root)
    if recorded != current:
        msg = "Inference implementation changed after the frozen run was initialized."
        raise ValueError(msg)
    return current


def _verify_run_implementation(paths: RunPaths) -> dict[str, str]:
    current = _verify_recorded_source_hashes(paths)
    manifest = _read_json(paths.output_root / "run_manifest.json")
    repo_root = Path(__file__).resolve().parents[1]
    repo_state = _git_snapshot(repo_root)
    recorded_git = manifest.get("git", {})
    if (
        not isinstance(recorded_git, dict)
        or recorded_git.get("dirty") is not False
        or repo_state["dirty"]
        or repo_state["head"] != recorded_git.get("head")
    ):
        msg = "Production inference requires the clean Git HEAD pinned by the run."
        raise ValueError(msg)
    return current


def _initialize_run(paths: RunPaths, *, allow_dirty: bool) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    resource_policy = {
        "default_jobs": safe_default_jobs(),
        "default_mutsig_jobs": min(2, safe_default_jobs()),
        "maximum_general_jobs": MAX_GENERAL_JOBS,
        "serial_canary_cohort": CANARY_COHORT,
        "memory_heavy_mutsig_cohorts": sorted(MEMORY_HEAVY_MUTSIG_COHORTS),
        "memory_heavy_mutsig_jobs": 1,
        "default_child_process_nice_increment": 10,
        "thread_environment": THREAD_LIMIT_ENV,
        "internal_task_environment": {INTERNAL_TASK_ENV: "1"},
        "prior_task_peak_rss_bytes": PRIOR_TASK_PEAK_RSS_BYTES,
        "memory_headroom_factor": MEMORY_HEADROOM_FACTOR,
        "minimum_available_memory_fraction": MIN_AVAILABLE_MEMORY_FRACTION,
        "minimum_free_disk_bytes": MIN_FREE_DISK_BYTES,
        "live_readback_before_every_wave": True,
        "live_readback_before_every_task": True,
    }
    expected = {
        "schema_version": SCHEMA_VERSION,
        "analysis": "tcga-revision-k500",
        "top_k": TOP_K,
        "cohorts": list(TCGA_COHORTS),
        "bmrs": list(BMRS),
        "source_root": paths.source_root.as_posix(),
        "mutsig_root": paths.mutsig_root.as_posix(),
        "feature_policy": "shared native-support count-ranked axis",
        "same_base_pair_policy": "excluded before fitting and testing",
        "required_lrt_contract": REQUIRED_LRT_CONTRACT,
        "required_pair_fit_contract": REQUIRED_PAIR_FIT_CONTRACT,
        "required_pair_fit_kkt_tolerance": REQUIRED_PAIR_FIT_KKT_TOL,
        "required_rho_contract": REQUIRED_RHO_CONTRACT,
        "undefined_rho_lrt_tolerance": REQUIRED_UNDEFINED_RHO_LRT_TOL,
        "required_contingency_table_contract": REQUIRED_CONTINGENCY_TABLE_CONTRACT,
        "required_log_odds_ratio_contract": REQUIRED_LOG_ODDS_RATIO_CONTRACT,
        "observation_support_universe": OBSERVATION_SUPPORT_UNIVERSE,
        "required_gene_support_contract": REQUIRED_GENE_SUPPORT_CONTRACT,
        "sample_axis_contract": SAMPLE_AXIS_CONTRACT,
        "resource_policy": resource_policy,
    }
    manifest_path = paths.output_root / "run_manifest.json"
    git = _git_snapshot(repo_root)
    if not allow_dirty and git["dirty"]:
        msg = (
            "Production K=500 inference requires a clean Git tree; commit or "
            "otherwise resolve every tracked/untracked change before launch."
        )
        raise RuntimeError(msg)
    if paths.output_root.exists():
        if not manifest_path.is_file():
            msg = (
                f"Output root already exists without this runner's manifest; refusing "
                f"to reuse or overwrite it: {paths.output_root}"
            )
            raise FileExistsError(msg)
        manifest = _read_json(manifest_path)
        for key, value in expected.items():
            if manifest.get(key) != value:
                msg = f"Run manifest mismatch for {key!r}; use a fresh output root."
                raise ValueError(msg)
        if manifest.get("implementation_sha256") != _source_snapshot(repo_root):
            msg = "Run implementation hashes changed; use a fresh output root."
            raise ValueError(msg)
        recorded_git = manifest.get("git", {})
        if not allow_dirty and (
            not isinstance(recorded_git, dict)
            or recorded_git.get("dirty") is not False
            or recorded_git.get("head") != git["head"]
        ):
            msg = "Run was not initialized from the current clean Git HEAD."
            raise RuntimeError(msg)
        return manifest

    paths.output_root.mkdir(parents=True, exist_ok=False)
    manifest = {
        **expected,
        "created_at_utc": _utc_now(),
        "git": git,
        "implementation_sha256": _source_snapshot(repo_root),
    }
    _write_json_atomic(manifest_path, manifest)
    return manifest


def _require_corrected_lrt() -> tuple[str, str, str, str, str, str]:
    actual = getattr(interaction_module, "LRT_CONTRACT", None)
    if actual != REQUIRED_LRT_CONTRACT:
        msg = (
            "K=500 launch blocked: dialect.models.interaction.LRT_CONTRACT must be "
            f"{REQUIRED_LRT_CONTRACT!r}, found {actual!r}. Implement and test the "
            "constrained independence-null MLE before running this grid."
        )
        raise RuntimeError(msg)
    pair_fit_actual = getattr(interaction_module, "PAIR_FIT_CONTRACT", None)
    if pair_fit_actual != REQUIRED_PAIR_FIT_CONTRACT:
        msg = (
            "K=500 launch blocked: dialect.models.interaction.PAIR_FIT_CONTRACT "
            f"must be {REQUIRED_PAIR_FIT_CONTRACT!r}, found {pair_fit_actual!r}."
        )
        raise RuntimeError(msg)
    pair_fit_kkt_actual = getattr(interaction_module, "PAIR_FIT_KKT_TOL", None)
    if pair_fit_kkt_actual != REQUIRED_PAIR_FIT_KKT_TOL:
        msg = (
            "K=500 launch blocked: dialect.models.interaction.PAIR_FIT_KKT_TOL "
            f"must be {REQUIRED_PAIR_FIT_KKT_TOL!r}, found "
            f"{pair_fit_kkt_actual!r}."
        )
        raise RuntimeError(msg)
    rho_actual = getattr(interaction_module, "RHO_CONTRACT", None)
    if rho_actual != REQUIRED_RHO_CONTRACT:
        msg = (
            "K=500 launch blocked: dialect.models.interaction.RHO_CONTRACT must "
            f"be {REQUIRED_RHO_CONTRACT!r}, found {rho_actual!r}."
        )
        raise RuntimeError(msg)
    undefined_rho_lrt_tol_actual = getattr(
        interaction_module,
        "UNDEFINED_RHO_LRT_TOL",
        None,
    )
    if undefined_rho_lrt_tol_actual != REQUIRED_UNDEFINED_RHO_LRT_TOL:
        msg = (
            "K=500 launch blocked: "
            "dialect.models.interaction.UNDEFINED_RHO_LRT_TOL must be "
            f"{REQUIRED_UNDEFINED_RHO_LRT_TOL!r}, found "
            f"{undefined_rho_lrt_tol_actual!r}."
        )
        raise RuntimeError(msg)
    contingency_actual = getattr(
        interaction_module,
        "CONTINGENCY_TABLE_CONTRACT",
        None,
    )
    if contingency_actual != REQUIRED_CONTINGENCY_TABLE_CONTRACT:
        msg = (
            "K=500 launch blocked: "
            "dialect.models.interaction.CONTINGENCY_TABLE_CONTRACT must be "
            f"{REQUIRED_CONTINGENCY_TABLE_CONTRACT!r}, found "
            f"{contingency_actual!r}."
        )
        raise RuntimeError(msg)
    log_odds_actual = getattr(
        interaction_module,
        "LOG_ODDS_RATIO_CONTRACT",
        None,
    )
    if log_odds_actual != REQUIRED_LOG_ODDS_RATIO_CONTRACT:
        msg = (
            "K=500 launch blocked: "
            "dialect.models.interaction.LOG_ODDS_RATIO_CONTRACT must be "
            f"{REQUIRED_LOG_ODDS_RATIO_CONTRACT!r}, found {log_odds_actual!r}."
        )
        raise RuntimeError(msg)
    gene_support_actual = getattr(gene_module, "OBSERVATION_SUPPORT_CONTRACT", None)
    if gene_support_actual != REQUIRED_GENE_SUPPORT_CONTRACT:
        msg = (
            "K=500 launch blocked: dialect.models.gene.OBSERVATION_SUPPORT_CONTRACT "
            f"must be {REQUIRED_GENE_SUPPORT_CONTRACT!r}, found "
            f"{gene_support_actual!r}."
        )
        raise RuntimeError(msg)
    return (
        str(actual),
        str(pair_fit_actual),
        str(rho_actual),
        str(gene_support_actual),
        str(contingency_actual),
        str(log_odds_actual),
    )


def _task_pmfs(
    paths: RunPaths,
    task: Task,
    counts: pd.DataFrame,
    features: list[str],
) -> dict[str, Any]:
    cohort_dir = paths.source_root / task.cohort
    if task.bmr in {"cbase", "dig"}:
        filename = "bmr_pmfs.csv" if task.bmr == "cbase" else "bmr_pmfs.dig.csv"
        all_pmfs = _load_strict_pmfs(cohort_dir / filename)
        missing = [feature for feature in features if feature not in all_pmfs]
        if missing:
            msg = f"{task.bmr} lost native feature support: {missing[:5]}"
            raise ValueError(msg)
        return {feature: all_pmfs[feature] for feature in features}

    selected_counts = counts.loc[:, features]
    kmax = int(selected_counts.to_numpy().max(initial=0))
    return build_lambda_pmfs(
        features,
        selected_counts.index,
        paths.mutsig_root / task.cohort,
        None,
        kmax,
        allow_cbase_fallback=False,
        require_all_features=True,
        require_all_samples=True,
        lambda_floor=None,
    )


def _build_genes(
    counts: pd.DataFrame,
    features: Sequence[str],
    pmfs: dict[str, Any],
) -> dict[str, Gene]:
    genes = {
        feature: Gene(
            name=feature,
            samples=counts.index,
            counts=counts[feature].to_numpy(),
            bmr_pmf=pmfs[feature],
        )
        for feature in features
    }
    if list(genes) != list(features):
        msg = "Gene construction did not preserve the frozen feature order."
        raise ValueError(msg)
    return genes


def _pairwise_record(interaction: Interaction) -> dict[str, Any]:
    taus = (
        interaction.tau_00,
        interaction.tau_01,
        interaction.tau_10,
        interaction.tau_11,
    )
    contingency = interaction.compute_contingency_table()
    likelihood_ratio = interaction.likelihood_ratio
    if likelihood_ratio is None:
        likelihood_ratio = interaction.compute_likelihood_ratio(taus)
    return {
        "Gene A": interaction.gene_a.name,
        "Gene B": interaction.gene_b.name,
        "Tau_00": interaction.tau_00,
        "Tau_10": interaction.tau_10,
        "Tau_01": interaction.tau_01,
        "Tau_11": interaction.tau_11,
        "_00_": contingency[0, 0],
        "_10_": contingency[1, 0],
        "_01_": contingency[0, 1],
        "_11_": contingency[1, 1],
        "Tau_1X": interaction.tau_10 + interaction.tau_11,
        "Tau_X1": interaction.tau_01 + interaction.tau_11,
        "Rho": interaction.compute_rho_for_direction(taus, likelihood_ratio),
        "Log Odds Ratio": interaction.compute_log_odds_ratio(taus),
        "Likelihood Ratio": likelihood_ratio,
        "Wald Statistic": interaction.compute_wald_statistic(taus),
        "Fit Algorithm": interaction.fit_algorithm,
        "Fit Converged": interaction.fit_converged,
        "Fit Iterations": interaction.fit_iterations,
        "Fit Last LL Gain": interaction.fit_last_log_likelihood_gain,
        "Fit Fixed-Point Residual": interaction.fit_fixed_point_residual,
        "Fit KKT Residual": interaction.fit_kkt_residual,
        "Pair Fit Contract": REQUIRED_PAIR_FIT_CONTRACT,
        "Null Log Likelihood": interaction.null_log_likelihood,
        "Alternative Log Likelihood": interaction.alternative_log_likelihood,
        "LRT Contract": REQUIRED_LRT_CONTRACT,
    }


def _write_pairwise_results(
    path: Path,
    genes: dict[str, Gene],
    features: Sequence[str],
) -> int:
    """Fit and stream the pair table to bound memory at K=500."""
    rows = 0
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAIRWISE_COLUMNS)
        writer.writeheader()
        for feature_a, feature_b in iter_tested_pairs(features):
            interaction = Interaction(genes[feature_a], genes[feature_b])
            estimate_taus_for_each_interaction([interaction])
            writer.writerow(_pairwise_record(interaction))
            rows += 1
        handle.flush()
        os.fsync(handle.fileno())
    return rows


def _validate_pairwise_rho(
    raw_rho: str,
    csv_order_taus: Sequence[float],
    likelihood_ratio: float,
    pair: tuple[str, str],
) -> None:
    """Recompute rho and enforce its finite-or-degenerate-null contract."""
    tau_00, tau_10, tau_01, tau_11 = csv_order_taus
    expected_rho = interaction_module.compute_marshall_olkin_rho(
        [tau_00, tau_01, tau_10, tau_11],
    )
    if expected_rho is None:
        if raw_rho != "" or likelihood_ratio > REQUIRED_UNDEFINED_RHO_LRT_TOL:
            msg = f"Invalid undefined-rho boundary result for pair {pair}."
            raise ValueError(msg)
        return
    try:
        actual_rho = float(raw_rho)
    except (TypeError, ValueError) as error:
        msg = f"Missing or non-numeric rho for pair {pair}."
        raise ValueError(msg) from error
    if (
        not np.isfinite(actual_rho)
        or abs(actual_rho) > 1 + REQUIRED_UNDEFINED_RHO_LRT_TOL
        or actual_rho != expected_rho
    ):
        msg = f"Reported rho does not match fitted taus for pair {pair}."
        raise ValueError(msg)


def _validate_pairwise_output(path: Path, contract: dict[str, Any]) -> int:
    counts = _read_counts(Path(str(contract["inputs"]["counts"]["path"])))
    if (
        _sequence_sha256(str(sample) for sample in counts.index)
        != contract["samples"]["ordered_ids_sha256"]
        or len(counts) != int(contract["samples"]["count"])
    ):
        msg = "Frozen count matrix no longer matches the contracted sample axis."
        raise ValueError(msg)
    n_samples = len(counts)
    mutation_masks = {
        feature: int.from_bytes(
            np.packbits(
                counts[feature].to_numpy(dtype=np.int64) > 0,
                bitorder="little",
            ).tobytes(),
            "little",
        )
        for feature in contract["features"]
    }
    expected_pairs = iter_tested_pairs(contract["features"])
    digest = hashlib.sha256()
    row_count = 0
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != PAIRWISE_COLUMNS:
            msg = f"Unexpected pairwise result schema: {path}"
            raise ValueError(msg)
        for expected in expected_pairs:
            row = next(reader, None)
            if row is None:
                msg = "Pairwise output ended before the frozen pair universe."
                raise ValueError(msg)
            actual = (row["Gene A"], row["Gene B"])
            if actual != expected:
                msg = (
                    f"Pair universe/order mismatch at row {row_count}: "
                    f"expected={expected}, actual={actual}"
                )
                raise ValueError(msg)
            if _base_gene(actual[0]) == _base_gene(actual[1]):
                msg = f"Same-base pair escaped the exclusion policy: {actual}"
                raise ValueError(msg)
            try:
                taus = [
                    float(row[column])
                    for column in ("Tau_00", "Tau_10", "Tau_01", "Tau_11")
                ]
                likelihood_ratio = float(row["Likelihood Ratio"])
                contingency = [
                    float(row[column])
                    for column in ("_00_", "_10_", "_01_", "_11_")
                ]
                fit_iterations = float(row["Fit Iterations"])
                fit_last_gain = float(row["Fit Last LL Gain"])
                fit_fixed_point_residual = float(row["Fit Fixed-Point Residual"])
                fit_kkt_residual = float(row["Fit KKT Residual"])
                null_log_likelihood = float(row["Null Log Likelihood"])
                alternative_log_likelihood = float(
                    row["Alternative Log Likelihood"],
                )
            except (TypeError, ValueError) as error:
                msg = f"Non-numeric fitted result for pair {actual}: {error}"
                raise ValueError(msg) from error
            if (
                not np.isfinite(taus).all()
                or any(value < 0 or value > 1 for value in taus)
                or not np.isclose(sum(taus), 1.0, atol=1e-6)
            ):
                msg = f"Invalid fitted tau simplex for pair {actual}: {taus}"
                raise ValueError(msg)
            if not np.isfinite(likelihood_ratio) or likelihood_ratio < -1e-8:
                msg = f"Invalid likelihood ratio for pair {actual}: {likelihood_ratio}"
                raise ValueError(msg)
            _validate_pairwise_rho(row["Rho"], taus, likelihood_ratio, actual)
            tau_00, tau_10, tau_01, tau_11 = taus
            log_odds_defined = tau_00 * tau_11 > 0 and tau_01 * tau_10 > 0
            if log_odds_defined:
                expected_log_odds = float(
                    np.log((tau_00 * tau_11) / (tau_01 * tau_10)),
                )
                expected_wald = float(
                    expected_log_odds
                    / np.sqrt(
                        (1 / tau_00)
                        + (1 / tau_01)
                        + (1 / tau_10)
                        + (1 / tau_11),
                    ),
                )
                try:
                    actual_log_odds = float(row["Log Odds Ratio"])
                    actual_wald = float(row["Wald Statistic"])
                except (TypeError, ValueError) as error:
                    msg = f"Missing conventional LOR/Wald statistic for pair {actual}."
                    raise ValueError(msg) from error
                if (
                    not np.isfinite(actual_log_odds)
                    or not np.isfinite(actual_wald)
                    or actual_log_odds != expected_log_odds
                    or actual_wald != expected_wald
                ):
                    msg = (
                        "Conventional LOR/Wald orientation mismatch for pair "
                        f"{actual}."
                    )
                    raise ValueError(msg)
            elif row["Log Odds Ratio"] != "" or row["Wald Statistic"] != "":
                msg = f"Boundary LOR/Wald statistic must be blank for pair {actual}."
                raise ValueError(msg)
            if (
                row["Fit Algorithm"] != REQUIRED_PAIR_FIT_CONTRACT
                or row["Fit Converged"].strip().lower() != "true"
                or not np.isfinite(fit_iterations)
                or fit_iterations < 0
                or not fit_iterations.is_integer()
                or not np.isfinite(fit_last_gain)
                or fit_last_gain < -1e-8
                or not np.isfinite(fit_fixed_point_residual)
                or fit_fixed_point_residual < 0
                or fit_fixed_point_residual > 1.000001e-8
                or not np.isfinite(fit_kkt_residual)
                or fit_kkt_residual < 0
                or fit_kkt_residual > 1.000001e-8
                or not np.isfinite(null_log_likelihood)
                or not np.isfinite(alternative_log_likelihood)
                or alternative_log_likelihood < null_log_likelihood - 1e-8
                or not np.isclose(
                    likelihood_ratio,
                    2 * (alternative_log_likelihood - null_log_likelihood),
                    rtol=1e-8,
                    atol=1e-8,
                )
                or row["Pair Fit Contract"] != REQUIRED_PAIR_FIT_CONTRACT
                or row["LRT Contract"] != REQUIRED_LRT_CONTRACT
            ):
                msg = f"Invalid convergence/LRT provenance for pair {actual}."
                raise ValueError(msg)
            if (
                not np.isfinite(contingency).all()
                or any(
                    value < 0 or not float(value).is_integer()
                    for value in contingency
                )
                or int(sum(contingency)) != int(contract["samples"]["count"])
            ):
                msg = f"Invalid contingency table for pair {actual}: {contingency}"
                raise ValueError(msg)
            mask_a = mutation_masks[actual[0]]
            mask_b = mutation_masks[actual[1]]
            n_11 = (mask_a & mask_b).bit_count()
            n_10 = mask_a.bit_count() - n_11
            n_01 = mask_b.bit_count() - n_11
            n_00 = n_samples - n_10 - n_01 - n_11
            expected_contingency = [n_00, n_10, n_01, n_11]
            if contingency != expected_contingency:
                msg = (
                    f"Observed-cell semantics do not match the count matrix for pair "
                    f"{actual}: expected={expected_contingency}, actual={contingency}"
                )
                raise ValueError(msg)
            encoded = f"{actual[0]}\t{actual[1]}".encode()
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
            row_count += 1
        extra = next(reader, None)
        if extra is not None:
            msg = f"Pairwise output contains extra row after frozen universe: {extra}"
            raise ValueError(msg)
    expected_count = int(contract["pair_policy"]["row_count"])
    if row_count != expected_count:
        msg = f"Pair row count {row_count} does not match contract {expected_count}."
        raise ValueError(msg)
    expected_hash = contract["pair_policy"]["ordered_pair_sha256"]
    if digest.hexdigest() != expected_hash:
        msg = "Pairwise output pair-universe hash does not match the cohort contract."
        raise ValueError(msg)
    return row_count


def validate_task_output(
    task_dir: Path,
    contract: dict[str, Any],
    *,
    require_manifest: bool = True,
) -> dict[str, Any]:
    """Validate a complete task directory against its frozen cohort contract."""
    _require_exact_sample_axis(contract)
    _require_full_observation_support(contract)
    single_path = task_dir / "single_gene_results.csv"
    pairwise_path = task_dir / "pairwise_interaction_results.csv"
    single = pd.read_csv(single_path)
    actual_features = [str(value) for value in single["Gene Name"]]
    if actual_features != contract["features"]:
        msg = "Single-gene output does not preserve the exact frozen feature order."
        raise ValueError(msg)
    pi_values = pd.to_numeric(single["Pi"], errors="raise").to_numpy(dtype=float)
    if (
        not np.isfinite(pi_values).all()
        or (pi_values < 0).any()
        or (pi_values > 1).any()
    ):
        msg = "Single-gene output contains an invalid constrained marginal MLE."
        raise ValueError(msg)
    mle_iterations = pd.to_numeric(
        single["MLE Iterations"],
        errors="raise",
    ).to_numpy(dtype=float)
    mle_log_likelihood = pd.to_numeric(
        single["MLE Log Likelihood"],
        errors="raise",
    ).to_numpy(dtype=float)
    mle_converged = single["MLE Converged"].astype(str).str.lower()
    single_lrt = pd.to_numeric(
        single["Likelihood Ratio"],
        errors="raise",
    ).to_numpy(dtype=float)
    single_lrt_status = set(single["Single-Gene LRT Status"].astype(str))
    if (
        not mle_converged.eq("true").all()
        or not np.isfinite(mle_iterations).all()
        or (mle_iterations <= 0).any()
        or not np.equal(mle_iterations, np.floor(mle_iterations)).all()
        or not np.isfinite(mle_log_likelihood).all()
        or (single_lrt < 0).any()
        or not single_lrt_status
        <= {"finite", "infinite-passenger-null-zero-probability"}
        or (
            single["Single-Gene LRT Status"].eq("finite")
            & ~np.isfinite(single_lrt)
        ).any()
        or (
            single["Single-Gene LRT Status"].eq(
                "infinite-passenger-null-zero-probability",
            )
            & ~np.isposinf(single_lrt)
        ).any()
        or not single["LRT Contract"].eq(REQUIRED_LRT_CONTRACT).all()
    ):
        msg = "Single-gene convergence/LRT provenance is incomplete or invalid."
        raise ValueError(msg)
    row_count = _validate_pairwise_output(pairwise_path, contract)
    validation = {
        "features": len(actual_features),
        "ordered_features_sha256": _sequence_sha256(actual_features),
        "pairs": row_count,
        "ordered_pair_sha256": contract["pair_policy"]["ordered_pair_sha256"],
        "single_gene_sha256": _sha256(single_path),
        "pairwise_sha256": _sha256(pairwise_path),
    }
    if require_manifest:
        manifest = _read_json(task_dir / "task_manifest.json")
        if manifest.get("exit_status") != 0:
            msg = f"Completed task does not record exit_status=0: {task_dir}"
            raise ValueError(msg)
        _validate_task_resource_usage(manifest, task_dir)
        expected_provenance = {
            "lrt_contract": REQUIRED_LRT_CONTRACT,
            "pair_fit_contract": REQUIRED_PAIR_FIT_CONTRACT,
            "pair_fit_kkt_tolerance": REQUIRED_PAIR_FIT_KKT_TOL,
            "rho_contract": REQUIRED_RHO_CONTRACT,
            "undefined_rho_lrt_tolerance": REQUIRED_UNDEFINED_RHO_LRT_TOL,
            "contingency_table_contract": REQUIRED_CONTINGENCY_TABLE_CONTRACT,
            "log_odds_ratio_contract": REQUIRED_LOG_ODDS_RATIO_CONTRACT,
            "observation_support_universe": OBSERVATION_SUPPORT_UNIVERSE,
            "gene_support_contract": REQUIRED_GENE_SUPPORT_CONTRACT,
            "sample_axis_contract": SAMPLE_AXIS_CONTRACT,
        }
        if any(
            manifest.get(key) != value
            for key, value in expected_provenance.items()
        ):
            msg = f"Task statistical-contract provenance is invalid: {task_dir}"
            raise ValueError(msg)
        if manifest.get("contract_sha256") != _json_sha256(contract):
            msg = f"Task contract hash mismatch: {task_dir}"
            raise ValueError(msg)
        if manifest.get("validation") != validation:
            msg = f"Task validation record drifted from its outputs: {task_dir}"
            raise ValueError(msg)
    return validation


def execute_task(
    paths: RunPaths,
    task: Task,
    *,
    nice_increment: int = 10,
    top_k: int = TOP_K,
    expected_contract_sha256: str | None = None,
) -> str:
    """Execute one task in an isolated staging directory and atomically publish it."""
    if task.cohort not in TCGA_COHORTS or task.bmr not in BMRS:
        msg = f"Invalid task: {task}"
        raise ValueError(msg)
    if nice_increment < 0:
        msg = "Task niceness increment must be nonnegative."
        raise ValueError(msg)
    if top_k == TOP_K:
        if expected_contract_sha256 is None:
            msg = "Production tasks require the orchestrator's frozen contract hash."
            raise ValueError(msg)
        _require_internal_task_environment()
        _require_live_resource_gate(
            paths,
            jobs=1,
            label=f"task-start-{task.cohort}-{task.bmr}",
        )
    (
        lrt_contract,
        pair_fit_contract,
        rho_contract,
        gene_support_contract,
        contingency_table_contract,
        log_odds_ratio_contract,
    ) = _require_corrected_lrt()
    implementation_sha256 = (
        _verify_run_implementation(paths) if top_k == TOP_K else {}
    )
    if nice_increment > 0:
        os.nice(nice_increment)
    contract = _load_verified_contract(paths, task.cohort, top_k=top_k)
    if top_k == TOP_K:
        _require_canonical_mutsig_maf_binding(contract)
    contract_sha256 = _json_sha256(contract)
    if (
        expected_contract_sha256 is not None
        and contract_sha256 != expected_contract_sha256
    ):
        msg = f"Orchestrator/task contract hash mismatch for {task.cohort}."
        raise ValueError(msg)
    final_dir = _task_dir(paths, task)
    if final_dir.exists():
        validate_task_output(final_dir, contract)
        return "already-complete"

    attempt_id = uuid.uuid4().hex
    work_dir = paths.output_root / "work" / task.cohort / f"{task.bmr}.{attempt_id}"
    work_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        counts = _read_counts(paths.source_root / task.cohort / "count_matrix.csv")
        features = [str(feature) for feature in contract["features"]]
        counts = counts.loc[:, features]
        observed_kmax = int(counts.to_numpy().max(initial=0))
        if observed_kmax != contract["mutsig_pmf_contract"][
            "selected_observed_count_max"
        ]:
            msg = "Selected observed count support changed after preflight."
            raise ValueError(msg)  # noqa: TRY301
        pmfs = _task_pmfs(paths, task, counts, features)
        if list(pmfs) != features:
            msg = "BMR PMFs do not preserve the exact frozen feature order."
            raise ValueError(msg)  # noqa: TRY301
        genes = _build_genes(counts, features, pmfs)
        estimate_pi_for_each_gene(genes.values())
        create_single_gene_results(
            list(genes.values()),
            str(work_dir / "single_gene_results.csv"),
            cbase_phi_vals_present=False,
        )
        written = _write_pairwise_results(
            work_dir / "pairwise_interaction_results.csv",
            genes,
            features,
        )
        if written != contract["pair_policy"]["row_count"]:
            msg = "Writer did not emit the complete frozen pair universe."
            raise ValueError(msg)  # noqa: TRY301
        validation = validate_task_output(
            work_dir,
            contract,
            require_manifest=False,
        )
        resource_usage = _task_resource_usage(started)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "cohort": task.cohort,
            "bmr": task.bmr,
            "top_k": top_k,
            "contract_sha256": contract_sha256,
            "native_support_only": True,
            "mutsig_cbase_feature_fallback": False,
            "same_base_pairs_excluded_before_fit": True,
            "lrt_contract": lrt_contract,
            "pair_fit_contract": pair_fit_contract,
            "pair_fit_kkt_tolerance": REQUIRED_PAIR_FIT_KKT_TOL,
            "rho_contract": rho_contract,
            "undefined_rho_lrt_tolerance": REQUIRED_UNDEFINED_RHO_LRT_TOL,
            "contingency_table_contract": contingency_table_contract,
            "log_odds_ratio_contract": log_odds_ratio_contract,
            "observation_support_universe": OBSERVATION_SUPPORT_UNIVERSE,
            "gene_support_contract": gene_support_contract,
            "sample_axis_contract": SAMPLE_AXIS_CONTRACT,
            "exit_status": 0,
            "completed_at_utc": _utc_now(),
            "resource_usage": resource_usage,
            "implementation_sha256": implementation_sha256,
            "validation": validation,
        }
        _write_json_atomic(work_dir / "task_manifest.json", manifest)
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        if final_dir.exists():
            msg = f"Refusing to overwrite a concurrently completed task: {final_dir}"
            raise FileExistsError(msg)  # noqa: TRY301
        work_dir.rename(final_dir)
        validate_task_output(final_dir, contract)
    except Exception:
        failure_traceback = traceback.format_exc()
        failure = {
            "schema_version": SCHEMA_VERSION,
            "cohort": task.cohort,
            "bmr": task.bmr,
            "attempt_id": attempt_id,
            "exit_status": 1,
            "failed_at_utc": _utc_now(),
            "resource_usage": _task_resource_usage(started),
            "traceback": failure_traceback,
        }
        _write_json_atomic(work_dir / "failure_manifest.json", failure)
        raise
    return "completed"


def safe_default_jobs(logical_cores: int | None = None) -> int:
    """Return at most three jobs and below half when the host has at least 3 cores."""
    cores = logical_cores if logical_cores is not None else (os.cpu_count() or 1)
    return max(1, min(MAX_GENERAL_JOBS, (cores - 1) // 2))


def _parse_darwin_memory_pressure(output: str) -> tuple[int, int]:
    total_match = re.search(r"The system has (\d+) ", output)
    free_match = re.search(r"System-wide memory free percentage: (\d+)%", output)
    if total_match is None or free_match is None:
        msg = "Could not parse macOS memory_pressure aggregate readback."
        raise RuntimeError(msg)
    total = int(total_match.group(1))
    free_percent = int(free_match.group(1))
    if total <= 0 or not 0 <= free_percent <= 100:
        msg = "macOS memory_pressure returned invalid aggregate values."
        raise RuntimeError(msg)
    return total, total * free_percent // 100


def _parse_linux_meminfo(content: str) -> tuple[int, int]:
    fields: dict[str, int] = {}
    for line in content.splitlines():
        name, separator, raw_value = line.partition(":")
        if not separator:
            continue
        pieces = raw_value.split()
        if len(pieces) == 2 and pieces[0].isdigit() and pieces[1] == "kB":
            fields[name] = int(pieces[0]) * 1024
    try:
        total = fields["MemTotal"]
        available = fields["MemAvailable"]
    except KeyError as error:
        msg = "Linux /proc/meminfo lacks MemTotal or MemAvailable."
        raise RuntimeError(msg) from error
    if total <= 0 or not 0 <= available <= total:
        msg = "Linux /proc/meminfo returned invalid aggregate values."
        raise RuntimeError(msg)
    return total, available


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path.resolve()
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            msg = f"No existing parent is available for disk readback: {path}"
            raise FileNotFoundError(msg)
        candidate = parent
    return candidate


def read_host_resources(output_root: Path) -> HostResourceSnapshot:
    """Read aggregate host memory, core, and target-filesystem capacity."""
    if sys.platform == "darwin":
        result = subprocess.run(
            ["/usr/bin/memory_pressure", "-Q"],
            check=True,
            capture_output=True,
            text=True,
        )
        total_memory, available_memory = _parse_darwin_memory_pressure(
            result.stdout,
        )
        memory_source = "/usr/bin/memory_pressure -Q"
    elif sys.platform.startswith("linux"):
        total_memory, available_memory = _parse_linux_meminfo(
            Path("/proc/meminfo").read_text(encoding="utf-8"),
        )
        memory_source = "/proc/meminfo MemAvailable"
    else:
        msg = f"Unsupported platform for live memory gating: {sys.platform!r}."
        raise RuntimeError(msg)
    disk_parent = _nearest_existing_parent(output_root)
    return HostResourceSnapshot(
        measured_at_utc=_utc_now(),
        logical_cores=os.cpu_count() or 1,
        total_memory_bytes=total_memory,
        available_memory_bytes=available_memory,
        free_disk_bytes=shutil.disk_usage(disk_parent).free,
        memory_source=memory_source,
    )


def evaluate_host_resource_gate(
    snapshot: HostResourceSnapshot,
    *,
    jobs: int,
) -> dict[str, Any]:
    """Evaluate the frozen half-machine launch policy against one readback."""
    reasons: list[str] = []
    if snapshot.logical_cores <= 0:
        reasons.append("logical core count must be positive")
    if snapshot.total_memory_bytes <= 0:
        reasons.append("total memory must be positive")
    if not 0 <= snapshot.available_memory_bytes <= snapshot.total_memory_bytes:
        reasons.append("available memory is outside the physical-memory range")
    if snapshot.free_disk_bytes < 0:
        reasons.append("free disk cannot be negative")
    try:
        measured_at = datetime.fromisoformat(snapshot.measured_at_utc)
        timestamp_is_utc = (
            measured_at.tzinfo is not None
            and measured_at.utcoffset() == UTC.utcoffset(None)
        )
    except (TypeError, ValueError):
        timestamp_is_utc = False
    if not timestamp_is_utc or not snapshot.memory_source:
        reasons.append("resource readback provenance is incomplete")

    required_by_prior_rss = math.ceil(
        jobs * PRIOR_TASK_PEAK_RSS_BYTES * MEMORY_HEADROOM_FACTOR,
    )
    required_by_fraction = math.ceil(
        max(snapshot.total_memory_bytes, 0) * MIN_AVAILABLE_MEMORY_FRACTION,
    )
    required_available = max(required_by_prior_rss, required_by_fraction)
    safe_cap = safe_default_jobs(max(snapshot.logical_cores, 1))
    if jobs <= 0 or jobs > safe_cap:
        reasons.append(f"jobs={jobs} exceeds safe live cap={safe_cap}")
    if (
        snapshot.total_memory_bytes > 0
        and 0 <= snapshot.available_memory_bytes <= snapshot.total_memory_bytes
        and snapshot.available_memory_bytes < required_available
    ):
        reasons.append(
            "available memory is below the prior-RSS/fraction headroom gate",
        )
    if snapshot.free_disk_bytes < MIN_FREE_DISK_BYTES:
        reasons.append("free disk is below the 2x historical-output gate")
    return {
        "passed": not reasons,
        "jobs": jobs,
        "safe_job_cap": safe_cap,
        "required_available_memory_bytes": required_available,
        "required_by_prior_rss_bytes": required_by_prior_rss,
        "required_by_fraction_bytes": required_by_fraction,
        "minimum_free_disk_bytes": MIN_FREE_DISK_BYTES,
        "reasons": reasons,
    }


def _require_live_resource_gate(
    paths: RunPaths,
    *,
    jobs: int,
    label: str,
) -> None:
    snapshot = read_host_resources(paths.output_root)
    evaluation = evaluate_host_resource_gate(snapshot, jobs=jobs)
    record = {
        "schema_version": SCHEMA_VERSION,
        "label": label,
        "snapshot": asdict(snapshot),
        "evaluation": evaluation,
    }
    record_path = (
        paths.output_root / "resource_readbacks" / f"{uuid.uuid4().hex}.json"
    )
    _write_json_atomic(record_path, record)
    if (
        snapshot.total_memory_bytes > 0
        and 0 <= snapshot.available_memory_bytes <= snapshot.total_memory_bytes
    ):
        available_memory = (
            f"{snapshot.available_memory_bytes / snapshot.total_memory_bytes:.0%}"
        )
    else:
        available_memory = "invalid"
    print(
        f"resource gate {label}: jobs={jobs}, "
        f"available_memory={available_memory}, "
        f"free_disk={snapshot.free_disk_bytes / _GIB:.1f} GiB",
    )
    if not evaluation["passed"]:
        msg = f"Live resource gate failed: {evaluation['reasons']}"
        raise RuntimeError(msg)


def _require_internal_task_environment() -> None:
    """Require the exact deterministic, single-threaded child launch environment."""
    required = {
        **THREAD_LIMIT_ENV,
        INTERNAL_TASK_ENV: "1",
        "PYTHONHASHSEED": "0",
    }
    mismatches = {
        name: os.environ.get(name)
        for name, expected in required.items()
        if os.environ.get(name) != expected
    }
    if mismatches:
        msg = (
            "Production internal tasks require the orchestrator's exact "
            f"single-thread environment; mismatches: {sorted(mismatches)}"
        )
        raise RuntimeError(msg)


def _validate_cli_resource_options(
    *,
    jobs: int,
    mutsig_jobs: int,
    nice_increment: int,
    logical_cores: int | None = None,
) -> None:
    """Fail closed on resource-related public CLI overrides."""
    cores = logical_cores if logical_cores is not None else (os.cpu_count() or 1)
    safe_cap = safe_default_jobs(cores)
    if jobs <= 0:
        msg = "--jobs must be positive."
        raise ValueError(msg)
    if jobs > safe_cap:
        msg = (
            f"--jobs {jobs} exceeds the safe cap {safe_cap} for "
            f"{cores} logical cores."
        )
        raise ValueError(msg)
    mutsig_cap = min(2, safe_cap)
    if mutsig_jobs <= 0 or mutsig_jobs > mutsig_cap:
        msg = (
            f"--mutsig-jobs must be between 1 and {mutsig_cap} for "
            f"{cores} logical cores."
        )
        raise ValueError(msg)
    if nice_increment < 0:
        msg = "--nice must be nonnegative."
        raise ValueError(msg)


def _parse_cohorts(value: str | None) -> list[str]:
    if value is None:
        return list(TCGA_COHORTS)
    cohorts = [item.strip().upper() for item in value.split(",") if item.strip()]
    unknown = sorted(set(cohorts) - set(TCGA_COHORTS))
    if unknown or not cohorts or len(cohorts) != len(set(cohorts)):
        msg = f"Invalid or duplicate TCGA cohort selection: {unknown or cohorts}"
        raise ValueError(msg)
    return cohorts


def _record_attempt(  # noqa: PLR0913
    paths: RunPaths,
    task: Task,
    attempt_id: str,
    command: list[str],
    return_code: int,
    log_path: Path,
    started_at: str,
) -> None:
    record = {
        "schema_version": SCHEMA_VERSION,
        "cohort": task.cohort,
        "bmr": task.bmr,
        "attempt_id": attempt_id,
        "started_at_utc": started_at,
        "finished_at_utc": _utc_now(),
        "command": command,
        "exit_status": return_code,
        "log": _file_record(log_path),
    }
    path = (
        paths.output_root
        / "attempts"
        / task.cohort
        / task.bmr
        / f"{attempt_id}.json"
    )
    _write_json_atomic(path, record)


def _invoke_task(paths: RunPaths, task: Task, nice_increment: int) -> tuple[Task, int]:
    attempt_id = uuid.uuid4().hex
    attempt_dir = paths.output_root / "attempts" / task.cohort / task.bmr
    attempt_dir.mkdir(parents=True, exist_ok=True)
    log_path = attempt_dir / f"{attempt_id}.log"
    contract_sha256 = _json_sha256(_read_json(_contract_path(paths, task.cohort)))
    command = [
        sys.executable,
        "-m",
        "analysis.run_tcga_revision_k500",
        "--source-root",
        paths.source_root.as_posix(),
        "--mutsig-root",
        paths.mutsig_root.as_posix(),
        "--output-root",
        paths.output_root.as_posix(),
        "--internal-cohort",
        task.cohort,
        "--internal-bmr",
        task.bmr,
        "--internal-contract-sha256",
        contract_sha256,
        "--nice",
        str(nice_increment),
    ]
    env = os.environ.copy()
    env.update(THREAD_LIMIT_ENV)
    env[INTERNAL_TASK_ENV] = "1"
    env["PYTHONHASHSEED"] = "0"
    started_at = _utc_now()
    with log_path.open("x", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    _record_attempt(
        paths,
        task,
        attempt_id,
        command,
        completed.returncode,
        log_path,
        started_at,
    )
    return task, completed.returncode


def _status(paths: RunPaths, cohorts: Sequence[str]) -> dict[str, Any]:
    records: dict[str, dict[str, str]] = {}
    counts = {"complete": 0, "pending": 0, "invalid": 0}
    for cohort in cohorts:
        contract_path = _contract_path(paths, cohort)
        contract = _read_json(contract_path) if contract_path.exists() else None
        records[cohort] = {}
        for bmr in BMRS:
            task_dir = _task_dir(paths, Task(cohort, bmr))
            if not task_dir.exists():
                state = "pending"
            elif contract is None:
                state = "invalid"
            else:
                try:
                    validate_task_output(task_dir, contract)
                    state = "complete"
                except (FileNotFoundError, ValueError, KeyError, TypeError):
                    state = "invalid"
            records[cohort][bmr] = state
            counts[state] += 1
    return {"counts": counts, "tasks": records}


def _run_task_batch(
    paths: RunPaths,
    tasks: Sequence[Task],
    *,
    jobs: int,
    nice_increment: int,
) -> int:
    failures = 0
    if not tasks:
        return failures
    if jobs <= 0:
        msg = "Task-batch concurrency must be positive."
        raise ValueError(msg)
    for start in range(0, len(tasks), jobs):
        wave = tasks[start : start + jobs]
        wave_jobs = len(wave)
        wave_number = start // jobs + 1
        _require_live_resource_gate(
            paths,
            jobs=wave_jobs,
            label="batch-{}-{}".format(
                wave_number,
                "-".join(f"{task.cohort}/{task.bmr}" for task in wave),
            ),
        )
        with ThreadPoolExecutor(max_workers=wave_jobs) as executor:
            futures = {
                executor.submit(_invoke_task, paths, task, nice_increment): task
                for task in wave
            }
            for future in as_completed(futures):
                task, return_code = future.result()
                if return_code == 0:
                    print(f"complete {task.cohort}/{task.bmr}")
                else:
                    failures += 1
                    print(
                        f"failed {task.cohort}/{task.bmr}: exit={return_code}",
                        file=sys.stderr,
                    )
    return failures


def _run_serial_canaries(
    paths: RunPaths,
    tasks: Sequence[Task],
    *,
    nice_increment: int,
) -> int:
    for position, task in enumerate(tasks, start=1):
        _require_live_resource_gate(
            paths,
            jobs=1,
            label=f"canary-{position}-{task.cohort}-{task.bmr}",
        )
        completed_task, return_code = _invoke_task(paths, task, nice_increment)
        if return_code != 0:
            print(
                f"failed {completed_task.cohort}/{completed_task.bmr}: "
                f"exit={return_code}",
                file=sys.stderr,
            )
            return 1
        print(f"complete {completed_task.cohort}/{completed_task.bmr}")
    return 0


def _require_validated_canary_outputs(paths: RunPaths) -> None:
    """Require all frozen CHOL outputs before launching a non-canary task."""
    task_dirs = [
        _task_dir(paths, Task(CANARY_COHORT, bmr))
        for bmr in BMRS
    ]
    missing = [task_dir for task_dir in task_dirs if not task_dir.is_dir()]
    if missing:
        msg = (
            f"Validated {CANARY_COHORT} canaries for all backgrounds are required "
            "before any non-canary production task; include CHOL in this run or "
            "resume from a run with complete validated CHOL outputs. "
            f"Missing: {[path.as_posix() for path in missing]}"
        )
        raise RuntimeError(msg)
    try:
        contract = _load_verified_contract(paths, CANARY_COHORT, top_k=TOP_K)
        for task_dir in task_dirs:
            validate_task_output(task_dir, contract)
    except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
        msg = (
            f"Validated {CANARY_COHORT} canaries for all backgrounds are required "
            "before any non-canary production task; include CHOL in this run or "
            "resume from a run with complete validated CHOL outputs."
        )
        raise RuntimeError(msg) from error


def _orchestrate(  # noqa: PLR0913
    paths: RunPaths,
    cohorts: Sequence[str],
    *,
    jobs: int,
    mutsig_jobs: int,
    nice_increment: int,
    preflight_only: bool,
) -> int:
    _initialize_run(paths, allow_dirty=preflight_only)
    contracts = {}
    for cohort in cohorts:
        contracts[cohort] = _ensure_contract(paths, cohort)
        print(f"preflight {cohort}: valid K={TOP_K} contract")
    if preflight_only:
        _verify_recorded_source_hashes(paths)
        try:
            _require_corrected_lrt()
        except RuntimeError as error:
            print(f"preflight launch gate: {error}", file=sys.stderr)
            return 2
        print("preflight launch gate: corrected LRT contract present")
        try:
            for contract in contracts.values():
                _require_canonical_mutsig_maf_binding(contract)
        except RuntimeError as error:
            print(f"preflight launch gate: {error}", file=sys.stderr)
            return 2
        return 0

    _require_corrected_lrt()
    for contract in contracts.values():
        _require_canonical_mutsig_maf_binding(contract)
    pending = []
    for cohort in cohorts:
        contract = _read_json(_contract_path(paths, cohort))
        for bmr in BMRS:
            task = Task(cohort, bmr)
            task_dir = _task_dir(paths, task)
            if task_dir.exists():
                validate_task_output(task_dir, contract)
                print(f"skip {cohort}/{bmr}: validated complete")
            else:
                pending.append(task)
    if not pending:
        print(json.dumps(_status(paths, cohorts), indent=2, sort_keys=True))
        return 0

    canaries = [task for task in pending if task.cohort == CANARY_COHORT]
    non_canaries = [task for task in pending if task.cohort != CANARY_COHORT]
    canary_failures = _run_serial_canaries(
        paths,
        canaries,
        nice_increment=nice_increment,
    )
    if canary_failures:
        print("canary failed; full K=500 grid remains gated", file=sys.stderr)
        print(json.dumps(_status(paths, cohorts), indent=2, sort_keys=True))
        return 1
    if non_canaries:
        _require_validated_canary_outputs(paths)

    cohort_level = [task for task in non_canaries if task.bmr != "mutsig"]
    light_mutsig = [
        task
        for task in non_canaries
        if task.bmr == "mutsig" and task.cohort not in MEMORY_HEAVY_MUTSIG_COHORTS
    ]
    heavy_mutsig = [
        task
        for task in non_canaries
        if task.bmr == "mutsig" and task.cohort in MEMORY_HEAVY_MUTSIG_COHORTS
    ]
    failures = canary_failures + _run_task_batch(
        paths,
        cohort_level,
        jobs=jobs,
        nice_increment=nice_increment,
    )
    failures += _run_task_batch(
        paths,
        light_mutsig,
        jobs=mutsig_jobs,
        nice_increment=nice_increment,
    )
    # The five largest MutSig tensors/selected-feature PMF sets run serially on the
    # 24 GB development host. This sacrifices some wall time to avoid swap pressure.
    failures += _run_task_batch(
        paths,
        heavy_mutsig,
        jobs=1,
        nice_increment=nice_increment,
    )
    print(json.dumps(_status(paths, cohorts), indent=2, sort_keys=True))
    return int(failures > 0)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("output/pancan"))
    parser.add_argument("--mutsig-root", type=Path, default=Path("output/mutsigsrc"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--cohorts",
        help="Comma-separated TCGA cohort subset; default is the canonical 32.",
    )
    parser.add_argument("--jobs", type=int, default=safe_default_jobs())
    parser.add_argument(
        "--mutsig-jobs",
        type=int,
        default=min(2, safe_default_jobs()),
    )
    parser.add_argument("--nice", type=int, default=10)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument(
        "--internal-cohort",
        choices=TCGA_COHORTS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--internal-bmr",
        choices=BMRS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--internal-contract-sha256", help=argparse.SUPPRESS)
    return parser


def main() -> None:
    """Validate, resume, or execute the frozen revision grid."""
    args = _parser().parse_args()
    paths = RunPaths(
        source_root=args.source_root.resolve(),
        mutsig_root=args.mutsig_root.resolve(),
        output_root=args.output_root.resolve(),
    )
    if args.nice < 0:
        msg = "--nice must be nonnegative."
        raise ValueError(msg)
    internal_values = (
        args.internal_cohort,
        args.internal_bmr,
        args.internal_contract_sha256,
    )
    if any(value is not None for value in internal_values):
        if any(value is None for value in internal_values):
            msg = "All internal task coordinates and the contract hash are required."
            raise ValueError(msg)
        execute_task(
            paths,
            Task(args.internal_cohort, args.internal_bmr),
            nice_increment=args.nice,
            expected_contract_sha256=args.internal_contract_sha256,
        )
        return

    cohorts = _parse_cohorts(args.cohorts)
    if args.status:
        if not paths.output_root.exists():
            msg = f"Run output root does not exist: {paths.output_root}"
            raise FileNotFoundError(msg)
        print(json.dumps(_status(paths, cohorts), indent=2, sort_keys=True))
        return
    _validate_cli_resource_options(
        jobs=args.jobs,
        mutsig_jobs=args.mutsig_jobs,
        nice_increment=args.nice,
    )
    exit_status = _orchestrate(
        paths,
        cohorts,
        jobs=args.jobs,
        mutsig_jobs=args.mutsig_jobs,
        nice_increment=args.nice,
        preflight_only=args.preflight_only,
    )
    raise SystemExit(exit_status)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
