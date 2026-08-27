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

Launch after the LRT gate is implemented and reviewed::

    PYTHONPATH=src /opt/anaconda3/envs/dialect/bin/python \
      -m analysis.run_tcga_revision_k500 \
      --output-root output/revision_tcga_k500_2026-08-27 --jobs 5
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
import subprocess
import sys
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
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

SCHEMA_VERSION = "1.0.0"
TOP_K = 500
BMRS = ("cbase", "dig", "mutsig")
MEMORY_HEAVY_MUTSIG_COHORTS = {"BRCA", "CRAD", "LGG", "SKCM", "UCEC"}
CANARY_COHORT = "CHOL"
REQUIRED_LRT_CONTRACT = "driver-independence-constrained-mle-v1"
REQUIRED_PAIR_FIT_CONTRACT = "deterministic-simplex-coordinate-ascent-v1"
REQUIRED_PAIR_FIT_KKT_TOL = 1e-8
REQUIRED_RHO_CONTRACT = "marshall-olkin-finite-or-degenerate-null-v1"
REQUIRED_UNDEFINED_RHO_LRT_TOL = 1e-8
OBSERVATION_SUPPORT_UNIVERSE = "full-observation-support-common-universe-v1"
REQUIRED_GENE_SUPPORT_CONTRACT = "latent-state-union-v1"
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
    Path("src/dialect/models/gene.py"),
    Path("src/dialect/models/interaction.py"),
    Path("src/dialect/utils/identify.py"),
)
_HASH_CHUNK_BYTES = 1024 * 1024


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


def _mutsig_contract(
    mutsig_dir: Path,
    count_samples: Sequence[str],
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
    patient_position = {patient: position for position, patient in enumerate(patients)}
    missing = [sample for sample in count_samples if sample not in patient_position]
    if missing:
        msg = (
            "MutSig patient axis lacks count-matrix samples; cohort-mean fallback is "
            f"prohibited: {missing[:5]}"
        )
        raise ValueError(msg)
    mapping = [f"{sample}\t{patient_position[sample]}" for sample in count_samples]
    count_sample_set = set(count_samples)
    return set(genes), {
        "dimensions": fields,
        "sample_mapping": {
            "cohort_samples": len(count_samples),
            "matched_samples": len(count_samples),
            "extra_mutsig_samples": sum(
                patient not in count_sample_set for patient in patients
            ),
            "cohort_mean_fallback_samples": 0,
            "ordered_mapping_sha256": _sequence_sha256(mapping),
        },
        "files": {
            "lambda": _file_record(lambda_path),
            "metadata": _file_record(meta_path),
            "genes": _file_record(genes_path),
            "patients": _file_record(patients_path),
        },
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


def build_cohort_contract(
    paths: RunPaths,
    cohort: str,
    *,
    top_k: int = TOP_K,
) -> dict[str, Any]:
    """Build a deterministic fail-closed contract for one cohort."""
    cohort_dir = paths.source_root / cohort
    count_path = cohort_dir / "count_matrix.csv"
    cbase_path = cohort_dir / "bmr_pmfs.csv"
    dig_path = cohort_dir / "bmr_pmfs.dig.csv"
    mutsig_dir = paths.mutsig_root / cohort
    counts = _read_counts(count_path)
    cbase_pmfs = _load_strict_pmfs(cbase_path)
    dig_pmfs = _load_strict_pmfs(dig_path)
    cbase_features = set(cbase_pmfs)
    dig_features = set(dig_pmfs)
    mutsig_genes, mutsig = _mutsig_contract(
        mutsig_dir,
        [str(sample) for sample in counts.index],
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
            "ordered_ids_sha256": _sequence_sha256(str(x) for x in counts.index),
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
            "cbase": _file_record(cbase_path),
            "dig": _file_record(dig_path),
            "mutsig": mutsig,
        },
    }
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
    _require_full_observation_support(contract)
    inputs = contract["inputs"]
    _verify_file_record(inputs["counts"])
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
        "observation_support_universe": OBSERVATION_SUPPORT_UNIVERSE,
        "required_gene_support_contract": REQUIRED_GENE_SUPPORT_CONTRACT,
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
        "resource_policy": {
            "default_jobs": safe_default_jobs(),
            "default_mutsig_jobs": 2,
            "serial_canary_cohort": CANARY_COHORT,
            "memory_heavy_mutsig_cohorts": sorted(MEMORY_HEAVY_MUTSIG_COHORTS),
            "memory_heavy_mutsig_jobs": 1,
            "child_process_nice_increment": 10,
            "thread_environment": THREAD_LIMIT_ENV,
        },
    }
    _write_json_atomic(manifest_path, manifest)
    return manifest


def _require_corrected_lrt() -> tuple[str, str, str, str]:
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
            "observation_support_universe": OBSERVATION_SUPPORT_UNIVERSE,
            "gene_support_contract": REQUIRED_GENE_SUPPORT_CONTRACT,
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
    (
        lrt_contract,
        pair_fit_contract,
        rho_contract,
        gene_support_contract,
    ) = _require_corrected_lrt()
    implementation_sha256 = (
        _verify_run_implementation(paths) if top_k == TOP_K else {}
    )
    if nice_increment > 0:
        os.nice(nice_increment)
    contract = _load_verified_contract(paths, task.cohort, top_k=top_k)
    contract_sha256 = _json_sha256(contract)
    if top_k == TOP_K and expected_contract_sha256 is None:
        msg = "Production tasks require the orchestrator's frozen contract hash."
        raise ValueError(msg)
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
            "observation_support_universe": OBSERVATION_SUPPORT_UNIVERSE,
            "gene_support_contract": gene_support_contract,
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
    """Return a default strictly below half the available logical cores."""
    cores = logical_cores if logical_cores is not None else (os.cpu_count() or 1)
    return max(1, min(5, (cores - 1) // 2))


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
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(_invoke_task, paths, task, nice_increment): task
            for task in tasks
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
    for task in tasks:
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
    for cohort in cohorts:
        _ensure_contract(paths, cohort)
        print(f"preflight {cohort}: valid K={TOP_K} contract")
    if preflight_only:
        _verify_recorded_source_hashes(paths)
        try:
            _require_corrected_lrt()
        except RuntimeError as error:
            print(f"preflight launch gate: {error}", file=sys.stderr)
            return 2
        print("preflight launch gate: corrected LRT contract present")
        return 0

    _require_corrected_lrt()
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
    parser.add_argument("--mutsig-jobs", type=int, default=2)
    parser.add_argument("--nice", type=int, default=10)
    parser.add_argument("--allow-high-concurrency", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--internal-cohort", choices=TCGA_COHORTS)
    parser.add_argument("--internal-bmr", choices=BMRS)
    parser.add_argument("--internal-contract-sha256")
    return parser


def main() -> None:
    """Validate, resume, or execute the frozen revision grid."""
    args = _parser().parse_args()
    paths = RunPaths(
        source_root=args.source_root.resolve(),
        mutsig_root=args.mutsig_root.resolve(),
        output_root=args.output_root.resolve(),
    )
    if args.internal_cohort or args.internal_bmr:
        if not args.internal_cohort or not args.internal_bmr:
            msg = "Both internal task coordinates are required."
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
    safe_cap = safe_default_jobs()
    if args.jobs <= 0:
        msg = "--jobs must be positive."
        raise ValueError(msg)
    if args.mutsig_jobs <= 0 or args.mutsig_jobs > 2:
        msg = "--mutsig-jobs must be 1 or 2 on the 24 GB development host."
        raise ValueError(msg)
    if args.jobs > safe_cap and not args.allow_high_concurrency:
        msg = (
            f"--jobs {args.jobs} exceeds the safe cap {safe_cap} for "
            f"{os.cpu_count() or 1} logical cores; pass --allow-high-concurrency "
            "only after explicit approval."
        )
        raise ValueError(msg)
    if args.nice < 0:
        msg = "--nice must be nonnegative."
        raise ValueError(msg)
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
