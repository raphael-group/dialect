"""Summarize focused K=500 provider results and diagnose PAAD/LAML q-values."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd

from analysis.postprocess_tcga_revision_focused import (
    RESULT_NAME,
    ROOT_MANIFEST_NAME,
    validate_derived_root,
)
from analysis.prepare_tcga_revision_focused import _parse_cohorts
from analysis.run_tcga_revision_k500 import BMRS

if TYPE_CHECKING:
    from collections.abc import Sequence

SCHEMA_VERSION: Final = "1.0.0"
DIAGNOSTIC_CONTRACT: Final = "focused-provider-result-diagnostics-v1"
THRESHOLDS: Final = (0.01, 0.05, 0.1, 0.2)
FOCAL_COHORTS: Final = ("LAML", "PAAD")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _provider_record(
    frame: pd.DataFrame,
    cohort: str,
    provider: str,
    threshold: float,
) -> dict[str, int | float | str | bool]:
    p_values = frame[f"{provider}_p_value"].to_numpy(dtype=np.float64)
    q_values = frame[f"{provider}_q_value"].to_numpy(dtype=np.float64)
    directions = frame[f"{provider}_direction"].astype("string")
    significant = q_values <= threshold
    return {
        "cohort": cohort,
        "provider": provider,
        "threshold": threshold,
        "pair_count": len(frame),
        "min_p_value": float(p_values.min()),
        "min_q_value": float(q_values.min()),
        "all_q_values_one": bool((q_values == 1).all()),
        "significant_count": int(significant.sum()),
        "me_count": int((significant & directions.eq("ME").to_numpy()).sum()),
        "co_count": int((significant & directions.eq("CO").to_numpy()).sum()),
        "unavailable_direction_count": int(
            (significant & directions.eq("unavailable").to_numpy()).sum(),
        ),
    }


def diagnose(
    *,
    postprocess_root: Path,
    output_root: Path,
    cohorts: Sequence[str],
) -> Path:
    """Write compact provider counts, overlap, and PAAD/LAML top-pair tables."""
    source_manifest = postprocess_root / ROOT_MANIFEST_NAME
    if not source_manifest.is_file():
        msg = "Focused postprocess manifest is missing."
        raise FileNotFoundError(msg)
    validate_derived_root(postprocess_root, cohorts)
    if output_root.exists() or output_root.is_symlink():
        msg = f"Refusing to overwrite diagnostic root: {output_root}"
        raise FileExistsError(msg)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.staging-",
            dir=output_root.parent,
        ),
    )
    count_records = []
    overlap_records = []
    focal_tables = []
    for cohort in cohorts:
        frame = pd.read_csv(
            postprocess_root / cohort / RESULT_NAME,
            float_precision="round_trip",
        )
        expected_columns = ["gene_a", "gene_b"]
        for provider in BMRS:
            expected_columns.extend(
                [
                    f"{provider}_likelihood_ratio",
                    f"{provider}_p_value",
                    f"{provider}_q_value",
                    f"{provider}_rho",
                    f"{provider}_direction",
                    f"{provider}_effect_identifiability",
                ],
            )
        bounded_columns = [
            f"{provider}_{suffix}"
            for provider in BMRS
            for suffix in ("p_value", "q_value")
        ]
        bounded = frame.loc[:, bounded_columns].to_numpy(dtype=np.float64)
        if (
            frame.columns.tolist() != expected_columns
            or frame[["gene_a", "gene_b"]].duplicated().any()
            or not np.isfinite(bounded).all()
            or (bounded < 0).any()
            or (bounded > 1).any()
        ):
            msg = f"Invalid focused postprocess table: {cohort}"
            raise ValueError(msg)
        for threshold in THRESHOLDS:
            masks = {}
            for provider in BMRS:
                count_records.append(
                    _provider_record(frame, cohort, provider, threshold),
                )
                masks[provider] = (
                    frame[f"{provider}_q_value"].to_numpy(dtype=np.float64)
                    <= threshold
                ) & frame[f"{provider}_direction"].isin(["ME", "CO"]).to_numpy()
            support = sum(mask.astype(np.int8) for mask in masks.values())
            overlap_records.append(
                {
                    "cohort": cohort,
                    "threshold": threshold,
                    "at_least_1_provider": int((support >= 1).sum()),
                    "at_least_2_providers": int((support >= 2).sum()),
                    "all_3_providers": int((support == 3).sum()),
                },
            )
        if cohort in FOCAL_COHORTS:
            for provider in BMRS:
                columns = [
                    "gene_a",
                    "gene_b",
                    f"{provider}_likelihood_ratio",
                    f"{provider}_p_value",
                    f"{provider}_q_value",
                    f"{provider}_rho",
                    f"{provider}_direction",
                    f"{provider}_effect_identifiability",
                ]
                top = frame.nsmallest(50, f"{provider}_p_value").loc[:, columns].copy()
                top.insert(0, "provider", provider)
                top.insert(0, "cohort", cohort)
                focal_tables.append(top)

    counts = pd.DataFrame(count_records)
    overlap = pd.DataFrame(overlap_records)
    focal = pd.concat(focal_tables, ignore_index=True)
    counts_path = staging / "cohort_provider_counts.csv"
    overlap_path = staging / "provider_overlap_counts.csv"
    focal_path = staging / "paad_laml_top_pairs.csv"
    counts.to_csv(counts_path, index=False, lineterminator="\n")
    overlap.to_csv(overlap_path, index=False, lineterminator="\n")
    focal.to_csv(focal_path, index=False, lineterminator="\n")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "contract": DIAGNOSTIC_CONTRACT,
        "postprocess_manifest_sha256": _sha256(source_manifest),
        "cohorts": list(cohorts),
        "thresholds": list(THRESHOLDS),
        "focal_cohorts": list(FOCAL_COHORTS),
        "interpretation": "descriptive-provider-overlap-not-formal-consensus",
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in (counts_path, overlap_path, focal_path)
        },
    }
    _write_atomic(
        staging / "diagnostic_manifest.json",
        _canonical_json(manifest) + b"\n",
    )
    staging.replace(output_root)
    return output_root


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--postprocess-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cohorts")
    return parser


def main() -> None:
    """Run focused result diagnostics."""
    args = _parser().parse_args()
    diagnose(
        postprocess_root=args.postprocess_root.resolve(),
        output_root=args.output_root.absolute(),
        cohorts=_parse_cohorts(args.cohorts),
    )


if __name__ == "__main__":
    main()
