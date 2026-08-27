"""Build zero-complete TCGA inputs on the exact MutSig patient axis.

The historical CBaSE count export contains only tumors with at least one retained
missense or nonsense event.  MutSig's ``persample_patients.txt`` is the canonical
mutation-profiled tumor axis and can additionally contain tumors whose modeled
counts are all zero.  This builder restores those rows, preserves every existing
count exactly, copies the CBaSE PMFs byte-for-byte, and regenerates DIG PMFs using
the corrected cohort size.

The output root is immutable: an existing path is never reused.  Files are first
written into an unpublished sibling staging directory and the complete tree is
published with one rename only after all cohort and root manifests validate.

Example::

    PYTHONPATH=src /opt/anaconda3/envs/dialect/bin/python \
      -m analysis.build_tcga_zero_complete_inputs \
      --out output/pancan_zero_complete_2026-08-27
"""

from __future__ import annotations

import argparse
import csv
import errno
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from dialect.bmr._dig_pmf import dig_results_to_bmr_pmfs

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

SCHEMA_VERSION = "1.0.0"
SAMPLE_AXIS_CONTRACT = "exact-ordered-mutsig-zero-complete-v1"
DIG_TAIL_EPS = 1e-7
_HASH_CHUNK_BYTES = 1024 * 1024
_INT64_MAX = np.iinfo(np.int64).max


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


def _sequence_sha256(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _matrix_values_sha256(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for sample, row in frame.iterrows():
        encoded_sample = str(sample).encode("utf-8")
        digest.update(len(encoded_sample).to_bytes(8, "big"))
        digest.update(encoded_sample)
        for value in row.to_numpy(dtype=np.int64):
            digest.update(int(value).to_bytes(8, "big", signed=True))
    return digest.hexdigest()


def _file_record(path: Path, *, display_path: str | None = None) -> dict[str, Any]:
    if not path.is_file():
        msg = f"Required file is missing: {path}"
        raise FileNotFoundError(msg)
    return {
        "path": display_path or path.resolve().as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_atomic(path: Path, payload: object) -> None:
    if path.exists():
        msg = f"Refusing to overwrite manifest: {path}"
        raise FileExistsError(msg)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(_canonical_json(payload))
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_count_matrix_atomic(frame: pd.DataFrame, path: Path) -> None:
    if path.exists():
        msg = f"Refusing to overwrite count matrix: {path}"
        raise FileExistsError(msg)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_patient_axis(path: Path) -> list[str]:
    if not path.is_file():
        msg = f"MutSig patient axis is missing: {path}"
        raise FileNotFoundError(msg)
    patients = path.read_text(encoding="utf-8").splitlines()
    if not patients or any(
        not patient or patient != patient.strip() for patient in patients
    ):
        msg = f"MutSig patient axis contains an empty or padded identifier: {path}"
        raise ValueError(msg)
    if len(patients) != len(set(patients)):
        msg = f"MutSig patient axis contains duplicate identifiers: {path}"
        raise ValueError(msg)
    return patients


def _read_integer_count_matrix(path: Path) -> pd.DataFrame:
    if not path.is_file():
        msg = f"Count matrix is missing: {path}"
        raise FileNotFoundError(msg)
    with path.open(encoding="utf-8", newline="") as handle:
        try:
            header = next(csv.reader(handle))
        except StopIteration as error:
            msg = f"Count matrix is empty: {path}"
            raise ValueError(msg) from error
    if len(header) < 2 or any(not feature for feature in header[1:]):
        msg = f"Count matrix must have an index and at least one named feature: {path}"
        raise ValueError(msg)
    if len(header[1:]) != len(set(header[1:])):
        msg = f"Count matrix contains duplicate feature names: {path}"
        raise ValueError(msg)

    index_name = header[0]
    frame = pd.read_csv(
        path,
        index_col=0,
        dtype={index_name: str},
        keep_default_na=False,
    )
    frame.index = pd.Index(frame.index.astype(str), name=frame.index.name)
    if frame.empty:
        msg = f"Count matrix must contain at least one sample: {path}"
        raise ValueError(msg)
    if frame.index.has_duplicates:
        msg = f"Count matrix contains duplicate sample identifiers: {path}"
        raise ValueError(msg)
    if any(not sample or sample != sample.strip() for sample in frame.index):
        msg = f"Count matrix contains an empty or padded sample identifier: {path}"
        raise ValueError(msg)

    try:
        numeric = frame.apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as error:
        msg = f"Count matrix contains a non-numeric value: {path}"
        raise ValueError(msg) from error
    values = numeric.to_numpy(dtype=float)
    if (
        not np.isfinite(values).all()
        or (values < 0).any()
        or (values > _INT64_MAX).any()
        or not np.equal(values, np.floor(values)).all()
    ):
        msg = f"Count matrix values must be finite nonnegative integers: {path}"
        raise ValueError(msg)
    result = numeric.astype(np.int64)
    result.index = frame.index
    result.columns = frame.columns
    return result


def _feature_summary(frame: pd.DataFrame) -> dict[str, Any]:
    totals = [(str(feature), int(frame[feature].sum())) for feature in frame.columns]
    return {
        "feature_count": len(frame.columns),
        "ordered_features_sha256": _sequence_sha256(str(x) for x in frame.columns),
        "feature_totals_sha256": _json_sha256(totals),
        "grand_total": sum(total for _, total in totals),
        "max_count": int(frame.to_numpy(dtype=np.int64).max()),
    }


def _validate_zero_completion(
    source: pd.DataFrame,
    completed: pd.DataFrame,
    patients: Sequence[str],
    inserted: Sequence[str],
) -> None:
    if list(completed.index) != list(patients):
        msg = "Completed count matrix does not exactly match the ordered MutSig axis"
        raise RuntimeError(msg)
    if list(completed.columns) != list(source.columns):
        msg = "Zero completion changed the count-matrix feature axis"
        raise RuntimeError(msg)
    if not completed.loc[source.index].equals(source):
        msg = "Zero completion changed one or more existing sample rows"
        raise RuntimeError(msg)
    if inserted and not (completed.loc[list(inserted)] == 0).to_numpy().all():
        msg = "One or more inserted samples is not an all-zero row"
        raise RuntimeError(msg)

    source_zero = source.index[(source == 0).all(axis=1)].astype(str).tolist()
    output_zero = completed.index[(completed == 0).all(axis=1)].astype(str).tolist()
    if source_zero:
        msg = (
            "The source count matrix already contains all-zero rows; only restored "
            f"MutSig-axis samples may be all zero: {source_zero[:5]}"
        )
        raise ValueError(msg)
    if output_zero != list(inserted):
        msg = "Completed all-zero rows are not exactly the inserted MutSig-axis samples"
        raise RuntimeError(msg)
    if _feature_summary(source) != _feature_summary(completed):
        msg = "Zero completion changed feature totals or observed-count support"
        raise RuntimeError(msg)


def _materialize_cbase(
    source: Path,
    destination: Path,
    *,
    mode: Literal["auto", "hardlink", "copy"],
) -> str:
    if destination.exists():
        msg = f"Refusing to overwrite CBaSE PMFs: {destination}"
        raise FileExistsError(msg)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    method = "copy"
    try:
        if mode != "copy":
            try:
                os.link(source, temporary)
                method = "hardlink"
            except OSError as error:
                if mode == "hardlink" or error.errno not in {
                    errno.EXDEV,
                    errno.EPERM,
                    errno.EACCES,
                    errno.ENOTSUP,
                }:
                    raise
        if not temporary.exists():
            shutil.copyfile(source, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(destination)
        _fsync_directory(destination.parent)
    finally:
        if temporary.exists():
            temporary.unlink()
    return method


def _write_dig_atomic(
    dig_results: Path,
    destination: Path,
    *,
    n_samples: int,
    max_count: int,
) -> None:
    if destination.exists():
        msg = f"Refusing to overwrite DIG PMFs: {destination}"
        raise FileExistsError(msg)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        dig_results_to_bmr_pmfs(
            str(dig_results),
            n_samples,
            str(temporary),
            max_count=max_count,
            tail_eps=DIG_TAIL_EPS,
        )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(destination)
        _fsync_directory(destination.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _git_output(repo_root: Path, arguments: Sequence[str]) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _git_provenance(repo_root: Path) -> dict[str, Any]:
    status = _git_output(
        repo_root,
        ["status", "--porcelain=v1", "--untracked-files=no"],
    ).splitlines()
    branch = _git_output(repo_root, ["branch", "--show-current"])
    return {
        "head": _git_output(repo_root, ["rev-parse", "HEAD"]),
        "branch": branch or None,
        "tracked_worktree_dirty": bool(status),
        "tracked_status": status,
        "tracked_status_sha256": _sequence_sha256(status),
    }


def _build_cohort(  # noqa: PLR0913
    cohort: str,
    *,
    source_root: Path,
    mutsig_root: Path,
    staging_root: Path,
    dig_results: Path,
    dig_record: dict[str, Any],
    git_provenance: dict[str, Any],
    cbase_mode: Literal["auto", "hardlink", "copy"],
) -> dict[str, Any]:
    source_dir = source_root / cohort
    mutsig_dir = mutsig_root / cohort
    source_count_path = source_dir / "count_matrix.csv"
    source_cbase_path = source_dir / "bmr_pmfs.csv"
    patients_path = mutsig_dir / "persample_patients.txt"
    source_records = {
        "count_matrix": _file_record(source_count_path),
        "cbase_pmfs": _file_record(source_cbase_path),
        "mutsig_patients": _file_record(patients_path),
        "dig_results": dig_record,
    }

    patients = _read_patient_axis(patients_path)
    source = _read_integer_count_matrix(source_count_path)
    patient_set = set(patients)
    extra_samples = [
        str(sample) for sample in source.index if sample not in patient_set
    ]
    if extra_samples:
        msg = (
            f"{cohort} count matrix has samples absent from the MutSig patient axis: "
            f"{extra_samples[:5]}"
        )
        raise ValueError(msg)
    source_sample_set = set(source.index.astype(str))
    inserted = [patient for patient in patients if patient not in source_sample_set]
    completed = source.reindex(patients, fill_value=0).astype(np.int64)
    completed.index.name = source.index.name
    _validate_zero_completion(source, completed, patients, inserted)

    output_dir = staging_root / cohort
    output_dir.mkdir()
    output_count_path = output_dir / "count_matrix.csv"
    output_cbase_path = output_dir / "bmr_pmfs.csv"
    output_dig_path = output_dir / "bmr_pmfs.dig.csv"

    _write_count_matrix_atomic(completed, output_count_path)
    serialized = _read_integer_count_matrix(output_count_path)
    _validate_zero_completion(source, serialized, patients, inserted)

    source_cbase_sha = source_records["cbase_pmfs"]["sha256"]
    cbase_method = _materialize_cbase(
        source_cbase_path,
        output_cbase_path,
        mode=cbase_mode,
    )
    if _sha256(source_cbase_path) != source_cbase_sha:
        msg = f"CBaSE source changed while materializing {cohort}"
        raise RuntimeError(msg)
    if _sha256(output_cbase_path) != source_cbase_sha:
        msg = f"CBaSE output is not byte-identical for {cohort}"
        raise RuntimeError(msg)

    source_feature_summary = _feature_summary(source)
    feature_summary = _feature_summary(serialized)
    _write_dig_atomic(
        dig_results,
        output_dig_path,
        n_samples=len(patients),
        max_count=feature_summary["max_count"],
    )
    if _sha256(dig_results) != dig_record["sha256"]:
        msg = "DIG results source changed during the build"
        raise RuntimeError(msg)
    if _sha256(source_count_path) != source_records["count_matrix"]["sha256"]:
        msg = f"Source count matrix changed while building {cohort}"
        raise RuntimeError(msg)
    if _sha256(patients_path) != source_records["mutsig_patients"]["sha256"]:
        msg = f"MutSig patient axis changed while building {cohort}"
        raise RuntimeError(msg)

    output_records = {
        "count_matrix": _file_record(
            output_count_path,
            display_path=f"{cohort}/count_matrix.csv",
        ),
        "cbase_pmfs": _file_record(
            output_cbase_path,
            display_path=f"{cohort}/bmr_pmfs.csv",
        ),
        "dig_pmfs": _file_record(
            output_dig_path,
            display_path=f"{cohort}/bmr_pmfs.dig.csv",
        ),
    }
    zero_rows = serialized.index[(serialized == 0).all(axis=1)].astype(str).tolist()
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "contract": SAMPLE_AXIS_CONTRACT,
        "cohort": cohort,
        "inputs": source_records,
        "outputs": output_records,
        "sample_axis": {
            "old_n": len(source),
            "new_n": len(serialized),
            "input_ordered_ids_sha256": _sequence_sha256(
                str(sample) for sample in source.index
            ),
            "mutsig_ordered_ids_sha256": _sequence_sha256(patients),
            "output_ordered_ids_sha256": _sequence_sha256(
                str(sample) for sample in serialized.index
            ),
            "inserted_count": len(inserted),
            "inserted_ids": inserted,
            "inserted_ids_sha256": _sequence_sha256(inserted),
            "source_zero_row_count": 0,
            "inserted_zero_row_count": len(inserted),
            "output_zero_row_count": len(zero_rows),
            "output_zero_ids_sha256": _sequence_sha256(zero_rows),
        },
        "count_matrix": {
            **feature_summary,
            "input_ordered_features_sha256": source_feature_summary[
                "ordered_features_sha256"
            ],
            "output_ordered_features_sha256": feature_summary[
                "ordered_features_sha256"
            ],
            "input_feature_totals_sha256": source_feature_summary[
                "feature_totals_sha256"
            ],
            "output_feature_totals_sha256": feature_summary[
                "feature_totals_sha256"
            ],
            "input_grand_total": source_feature_summary["grand_total"],
            "output_grand_total": feature_summary["grand_total"],
            "input_max_count": source_feature_summary["max_count"],
            "output_max_count": feature_summary["max_count"],
            "input_matrix_values_sha256": _matrix_values_sha256(source),
            "output_existing_rows_values_sha256": _matrix_values_sha256(
                serialized.loc[source.index],
            ),
            "output_matrix_values_sha256": _matrix_values_sha256(serialized),
        },
        "dig": {
            "converter": (
                "dialect.bmr._dig_pmf.dig_results_to_bmr_pmfs"
            ),
            "n_samples": len(patients),
            "max_count": feature_summary["max_count"],
            "tail_eps": DIG_TAIL_EPS,
        },
        "materialization": {"cbase_pmfs": cbase_method},
        "git": git_provenance,
    }
    if (
        manifest["count_matrix"]["input_matrix_values_sha256"]
        != manifest["count_matrix"]["output_existing_rows_values_sha256"]
    ):
        msg = f"Existing-row value hash changed for {cohort}"
        raise RuntimeError(msg)
    _write_json_atomic(output_dir / "input_manifest.json", manifest)
    return manifest


def _validate_cohorts(
    cohorts: Sequence[str] | None,
    mutsig_root: Path,
) -> tuple[str, ...]:
    selected = (
        list(cohorts)
        if cohorts is not None
        else [
            path.name
            for path in mutsig_root.iterdir()
            if path.is_dir() and (path / "persample_patients.txt").is_file()
        ]
    )
    if not selected:
        msg = "No cohorts were selected or discovered"
        raise ValueError(msg)
    if len(selected) != len(set(selected)):
        msg = "Cohort selection contains duplicates"
        raise ValueError(msg)
    if any(
        not cohort
        or cohort != Path(cohort).name
        or cohort in {".", ".."}
        for cohort in selected
    ):
        msg = "Cohort identifiers must be nonempty path basenames"
        raise ValueError(msg)
    return tuple(sorted(selected))


def build_zero_complete_inputs(  # noqa: PLR0913
    source_root: str | Path,
    mutsig_root: str | Path,
    dig_results: str | Path,
    out: str | Path,
    *,
    cohorts: Sequence[str] | None = None,
    cbase_mode: Literal["auto", "hardlink", "copy"] = "auto",
) -> Path:
    """Build and atomically publish immutable zero-complete cohort inputs."""
    source_root = Path(source_root).resolve()
    mutsig_root = Path(mutsig_root).resolve()
    dig_results = Path(dig_results).resolve()
    output_root = Path(out).resolve()
    if cbase_mode not in {"auto", "hardlink", "copy"}:
        msg = f"Unknown CBaSE materialization mode: {cbase_mode}"
        raise ValueError(msg)
    if os.path.lexists(output_root):
        msg = f"Refusing to reuse existing output root: {output_root}"
        raise FileExistsError(msg)
    if not source_root.is_dir() or not mutsig_root.is_dir():
        msg = "Source and MutSig roots must both exist as directories"
        raise FileNotFoundError(msg)
    selected = _validate_cohorts(cohorts, mutsig_root)

    repo_root = Path(__file__).resolve().parents[1]
    git_provenance = _git_provenance(repo_root)
    generator_record = _file_record(
        Path(__file__).resolve(),
        display_path="analysis/build_tcga_zero_complete_inputs.py",
    )
    dig_record = _file_record(dig_results)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.staging-",
            dir=output_root.parent,
        ),
    )
    published = False
    try:
        manifests = [
            _build_cohort(
                cohort,
                source_root=source_root,
                mutsig_root=mutsig_root,
                staging_root=staging_root,
                dig_results=dig_results,
                dig_record=dig_record,
                git_provenance=git_provenance,
                cbase_mode=cbase_mode,
            )
            for cohort in selected
        ]
        cohort_records = [
            {
                "cohort": cohort,
                "manifest_sha256": _sha256(
                    staging_root / cohort / "input_manifest.json",
                ),
            }
            for cohort in selected
        ]
        root_manifest = {
            "schema_version": SCHEMA_VERSION,
            "contract": SAMPLE_AXIS_CONTRACT,
            "cohorts": list(selected),
            "cohort_count": len(selected),
            "cohort_manifests": cohort_records,
            "cohort_manifests_sha256": _json_sha256(cohort_records),
            "sample_totals": {
                "old_n": sum(item["sample_axis"]["old_n"] for item in manifests),
                "new_n": sum(item["sample_axis"]["new_n"] for item in manifests),
                "inserted_count": sum(
                    item["sample_axis"]["inserted_count"] for item in manifests
                ),
                "output_zero_row_count": sum(
                    item["sample_axis"]["output_zero_row_count"]
                    for item in manifests
                ),
            },
            "inputs_sha256": _json_sha256(
                [
                    {
                        "cohort": item["cohort"],
                        "inputs": {
                            name: record["sha256"]
                            for name, record in item["inputs"].items()
                        },
                    }
                    for item in manifests
                ],
            ),
            "outputs_sha256": _json_sha256(
                [
                    {
                        "cohort": item["cohort"],
                        "outputs": {
                            name: record["sha256"]
                            for name, record in item["outputs"].items()
                        },
                    }
                    for item in manifests
                ],
            ),
            "dig_results": dig_record,
            "generator": generator_record,
            "git": git_provenance,
        }
        _write_json_atomic(staging_root / "input_manifest.json", root_manifest)
        if os.path.lexists(output_root):
            msg = f"Output root appeared during the build: {output_root}"
            raise FileExistsError(msg)
        staging_root.rename(output_root)
        published = True
        _fsync_directory(output_root.parent)
    finally:
        if not published and staging_root.exists():
            shutil.rmtree(staging_root)
    return output_root


def main() -> None:
    """Build corrected TCGA inputs from command-line paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("output/pancan"))
    parser.add_argument("--mutsig-root", type=Path, default=Path("output/mutsigsrc"))
    parser.add_argument(
        "--dig-results",
        type=Path,
        default=Path("external/DIGDriver/run/Pancan.genes.results.txt"),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cohorts", nargs="+")
    parser.add_argument(
        "--cbase-mode",
        choices=("auto", "hardlink", "copy"),
        default="auto",
    )
    args = parser.parse_args()
    result = build_zero_complete_inputs(
        args.source_root,
        args.mutsig_root,
        args.dig_results,
        args.out,
        cohorts=args.cohorts,
        cbase_mode=args.cbase_mode,
    )
    print(result)


if __name__ == "__main__":
    main()
