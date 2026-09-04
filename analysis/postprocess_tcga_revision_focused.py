"""Derive provider-specific p- and q-values for the focused K=500 grid.

The fit tests dependence once per unordered pair.  This stage therefore applies one
Benjamini-Hochberg correction to the complete within-cohort pair family for each BMR,
then labels direction from the fitted Marshall-Olkin rho.  It does not select a q
threshold or combine providers; those reporting decisions are frozen only after the
calibration and PAAD/LAML diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pandas as pd
from scipy.stats import chi2

from analysis import run_tcga_revision_focused as focused_runner
from analysis.prepare_tcga_revision_focused import _parse_cohorts
from analysis.run_tcga_revision_k500 import BMRS, PAIRWISE_COLUMNS

if TYPE_CHECKING:
    from collections.abc import Sequence

SCHEMA_VERSION: Final = "1.0.0"
DERIVATION_CONTRACT: Final = "focused-provider-complete-family-bh-v1"
ROOT_CONTRACT: Final = "focused-32x3-provider-inference-v1"
RESULT_NAME: Final = "provider_inference.csv"
COHORT_MANIFEST_NAME: Final = "cohort_manifest.json"
ROOT_MANIFEST_NAME: Final = "postprocess_manifest.json"


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


def _file_record(path: Path, *, relative_to: Path) -> dict[str, int | str]:
    if not path.is_file() or path.is_symlink():
        msg = f"Required regular file is missing or unsafe: {path}"
        raise FileNotFoundError(msg)
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


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


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Return stable BH adjusted p-values for one complete finite family."""
    values = np.asarray(p_values, dtype=np.float64)
    if (
        values.ndim != 1
        or not np.isfinite(values).all()
        or (values < 0).any()
        or (values > 1).any()
    ):
        msg = "BH requires a one-dimensional finite p-value family in [0, 1]."
        raise ValueError(msg)
    count = len(values)
    if count == 0:
        return values.copy()
    order = np.argsort(values, kind="stable")
    ranked = values[order]
    adjusted = ranked * count / np.arange(1, count + 1, dtype=np.float64)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result


def _direction(rho: pd.Series) -> pd.Series:
    values = pd.to_numeric(rho, errors="coerce").to_numpy(dtype=np.float64)
    labels = np.full(len(values), "unavailable", dtype=object)
    labels[np.isfinite(values) & (values < 0)] = "ME"
    labels[np.isfinite(values) & (values > 0)] = "CO"
    labels[np.isfinite(values) & (values == 0)] = "neutral"
    return pd.Series(labels, index=rho.index, dtype="string")


def _read_provider(run_root: Path, cohort: str, provider: str) -> pd.DataFrame:
    task_root = run_root / "tasks" / cohort / provider
    contract = json.loads(
        (run_root / "contracts" / f"{cohort}.json").read_text(encoding="utf-8"),
    )
    contract_sha256 = hashlib.sha256(_canonical_json(contract)).hexdigest()
    manifest = json.loads(
        (task_root / "task_manifest.json").read_text(encoding="utf-8"),
    )
    source = task_root / "pairwise_interaction_results.csv"
    source_record = manifest.get("outputs", {}).get(source.name, {})
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("contract") != focused_runner.TASK_CONTRACT
        or manifest.get("cohort") != cohort
        or manifest.get("provider") != provider
        or manifest.get("top_k") != 500
        or manifest.get("contract_sha256") != contract_sha256
        or manifest.get("config_sha256")
        != _sha256(focused_runner.CONFIG_PATH)
        or manifest.get("pairwise_rows") != contract["pair_policy"]["row_count"]
        or set(manifest.get("outputs", {}))
        != {"pairwise_interaction_results.csv", "single_gene_results.csv"}
        or {path.name for path in task_root.iterdir()}
        != {
            "pairwise_interaction_results.csv",
            "single_gene_results.csv",
            "task_manifest.json",
        }
        or source_record.get("bytes") != source.stat().st_size
        or source_record.get("sha256") != _sha256(source)
    ):
        msg = f"Task output is not bound by its manifest: {cohort}/{provider}"
        raise ValueError(msg)
    frame = pd.read_csv(source, float_precision="round_trip")
    if tuple(frame.columns) != PAIRWISE_COLUMNS:
        msg = f"Unexpected pairwise schema: {cohort}/{provider}"
        raise ValueError(msg)
    likelihood_ratio = frame["Likelihood Ratio"].to_numpy(dtype=np.float64)
    if (
        len(frame) != contract["pair_policy"]["row_count"]
        or not np.isfinite(likelihood_ratio).all()
        or (likelihood_ratio < -1e-10).any()
    ):
        msg = f"Invalid profile likelihood ratios: {cohort}/{provider}"
        raise ValueError(msg)
    likelihood_ratio = np.maximum(likelihood_ratio, 0.0)
    p_values = chi2.sf(likelihood_ratio, df=1)
    return pd.DataFrame(
        {
            "gene_a": frame["Gene A"].astype("string"),
            "gene_b": frame["Gene B"].astype("string"),
            f"{provider}_likelihood_ratio": likelihood_ratio,
            f"{provider}_p_value": p_values,
            f"{provider}_q_value": benjamini_hochberg(p_values),
            f"{provider}_rho": pd.to_numeric(frame["Rho"], errors="coerce"),
            f"{provider}_direction": _direction(frame["Rho"]),
            f"{provider}_effect_identifiability": frame[
                "Effect Identifiability"
            ].astype("string"),
        },
    )


def _validate_completion(
    run_root: Path,
    cohorts: Sequence[str],
) -> Path:
    completion = run_root / "completion_manifest.json"
    payload = json.loads(completion.read_text(encoding="utf-8"))
    expected_tasks = {(cohort, provider) for cohort in cohorts for provider in BMRS}
    observed_tasks = {
        (str(task.get("cohort")), str(task.get("provider")))
        for task in payload.get("tasks", [])
    }
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("contract") != focused_runner.COMPLETION_CONTRACT
        or payload.get("config_sha256") != _sha256(focused_runner.CONFIG_PATH)
        or payload.get("task_count") != len(payload.get("tasks", []))
        or len(observed_tasks) != len(payload.get("tasks", []))
        or not set(cohorts) <= set(payload.get("cohorts", []))
        or not expected_tasks <= observed_tasks
    ):
        msg = "Focused K=500 completion manifest does not cover the requested grid."
        raise ValueError(msg)
    for task in payload["tasks"]:
        coordinate = (str(task.get("cohort")), str(task.get("provider")))
        if coordinate not in expected_tasks:
            continue
        record = task.get("manifest", {})
        path = run_root / str(record.get("path", ""))
        if (
            record.get("path")
            != f"tasks/{coordinate[0]}/{coordinate[1]}/task_manifest.json"
            or record.get("bytes") != path.stat().st_size
            or record.get("sha256") != _sha256(path)
        ):
            msg = f"Completion task manifest changed: {coordinate[0]}/{coordinate[1]}"
            raise ValueError(msg)
    return completion


def validate_derived_root(
    output_root: Path,
    cohorts: Sequence[str],
) -> dict[str, Any]:
    """Validate receipt-bound derived tables before downstream consumption."""
    root_path = output_root / ROOT_MANIFEST_NAME
    payload = json.loads(root_path.read_text(encoding="utf-8"))
    records = {
        str(record.get("path")): record
        for record in payload.get("cohort_manifests", [])
    }
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("contract") != ROOT_CONTRACT
        or payload.get("cohort_count") != len(payload.get("cohorts", []))
        or payload.get("provider_family_count")
        != len(payload.get("cohorts", [])) * len(BMRS)
        or not set(cohorts) <= set(payload.get("cohorts", []))
        or len(records) != len(payload.get("cohort_manifests", []))
    ):
        msg = "Focused postprocess root manifest is invalid."
        raise ValueError(msg)
    for cohort in cohorts:
        relative_manifest = f"{cohort}/{COHORT_MANIFEST_NAME}"
        record = records.get(relative_manifest, {})
        manifest_path = output_root / relative_manifest
        if (
            record.get("bytes") != manifest_path.stat().st_size
            or record.get("sha256") != _sha256(manifest_path)
        ):
            msg = f"Focused cohort manifest changed: {cohort}"
            raise ValueError(msg)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        result_path = output_root / cohort / RESULT_NAME
        output = manifest.get("output", {})
        if (
            manifest.get("schema_version") != SCHEMA_VERSION
            or manifest.get("contract") != DERIVATION_CONTRACT
            or manifest.get("cohort") != cohort
            or manifest.get("providers") != list(BMRS)
            or output.get("path") != f"{cohort}/{RESULT_NAME}"
            or output.get("bytes") != result_path.stat().st_size
            or output.get("sha256") != _sha256(result_path)
            or {path.name for path in (output_root / cohort).iterdir()}
            != {RESULT_NAME, COHORT_MANIFEST_NAME}
        ):
            msg = f"Focused derived cohort output is invalid: {cohort}"
            raise ValueError(msg)
    return payload


def derive_cohort(run_root: Path, cohort: str, output_root: Path) -> dict[str, Any]:
    """Validate and derive all three provider families for one cohort."""
    frames = [_read_provider(run_root, cohort, provider) for provider in BMRS]
    pair_axis = frames[0].loc[:, ["gene_a", "gene_b"]]
    for provider, frame in zip(BMRS[1:], frames[1:], strict=True):
        if not pair_axis.equals(frame.loc[:, ["gene_a", "gene_b"]]):
            msg = f"Provider pair axes differ for {cohort}/{provider}."
            raise ValueError(msg)
    combined = pair_axis.copy()
    for frame in frames:
        combined = pd.concat([combined, frame.iloc[:, 2:]], axis=1)

    cohort_root = output_root / cohort
    cohort_root.mkdir(parents=True)
    result_path = cohort_root / RESULT_NAME
    combined.to_csv(result_path, index=False, lineterminator="\n")
    sources = {
        provider: _file_record(
            run_root
            / "tasks"
            / cohort
            / provider
            / "pairwise_interaction_results.csv",
            relative_to=run_root,
        )
        for provider in BMRS
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "contract": DERIVATION_CONTRACT,
        "cohort": cohort,
        "pair_count": len(combined),
        "providers": list(BMRS),
        "family": "all-matched-unordered-pairs-excluding-same-base-M:N",
        "multiplicity": "provider-specific-BH-over-complete-within-cohort-family",
        "direction": "rho-sign-after-nondirectional-profile-LRT",
        "reporting_threshold_selected": False,
        "sources": sources,
        "output": _file_record(result_path, relative_to=output_root),
    }
    _write_atomic(
        cohort_root / COHORT_MANIFEST_NAME,
        _canonical_json(manifest) + b"\n",
    )
    return manifest


def derive(
    *,
    run_root: Path,
    output_root: Path,
    cohorts: Sequence[str],
) -> Path:
    """Publish a no-replace, fully receipt-bound provider inference tree."""
    completion = _validate_completion(run_root, cohorts)
    if output_root.exists() or output_root.is_symlink():
        msg = f"Refusing to overwrite postprocess root: {output_root}"
        raise FileExistsError(msg)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.staging-",
            dir=output_root.parent,
        ),
    )
    manifests = [derive_cohort(run_root, cohort, staging) for cohort in cohorts]
    root = {
        "schema_version": SCHEMA_VERSION,
        "contract": ROOT_CONTRACT,
        "run_completion": _file_record(completion, relative_to=run_root),
        "cohorts": list(cohorts),
        "cohort_count": len(cohorts),
        "provider_family_count": len(cohorts) * len(BMRS),
        "pair_count_per_provider": sum(
            int(manifest["pair_count"]) for manifest in manifests
        ),
        "reporting_threshold_selected": False,
        "cohort_manifests": [
            _file_record(
                staging / cohort / COHORT_MANIFEST_NAME,
                relative_to=staging,
            )
            for cohort in cohorts
        ],
    }
    _write_atomic(
        staging / ROOT_MANIFEST_NAME,
        _canonical_json(root) + b"\n",
    )
    staging.replace(output_root)
    validate_derived_root(output_root, cohorts)
    return output_root


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cohorts")
    return parser


def main() -> None:
    """Run focused provider-specific postprocessing."""
    args = _parser().parse_args()
    derive(
        run_root=args.run_root.resolve(),
        output_root=args.output_root.absolute(),
        cohorts=_parse_cohorts(args.cohorts),
    )


if __name__ == "__main__":
    main()
