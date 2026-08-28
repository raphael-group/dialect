"""Materialize the frozen participant-unique TCGA revision sample axes.

This result-blind stage reads only commit-matched cBioPortal DataHub sequenced-case
lists. It does not read mutations, BMRs, or association results. Every source blob,
study identifier, sample count, participant projection, and ordered output axis is
verified against the frozen receipts in :mod:`dialect.data.tcga`.

The output tree is immutable. Cohort files are written beneath an unpublished sibling
staging directory and exposed with one rename only after all manifests validate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import dialect.data.tcga as tcga_data
from dialect.data.tcga import (
    PRIMARY_DISEASE_SAMPLE_TYPE_CODES,
    TCGA_CASE_LIST_RECEIPTS,
    TCGA_COHORTS,
    TCGA_DATAHUB_COMMIT,
    TCGA_DATAHUB_TREE,
    TCGA_SELECTED_SAMPLE_AXIS_SHA256,
    build_tcga_selected_sample_axis,
    tcga_datahub_case_list_path,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

SCHEMA_VERSION = "1.0.0"
POPULATION_CONTRACT = "pinned-datahub-participant-unique-sample-axis-v1"
_HASH_CHUNK_BYTES = 1024 * 1024
_PUBLISH_CLAIM_SUFFIX = ".publish-claim"


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


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


def _file_record(path: Path, *, display_path: str) -> dict[str, int | str]:
    if not path.is_file():
        msg = f"Required materialized file is missing: {path}"
        raise FileNotFoundError(msg)
    return {
        "path": display_path,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_bytes_atomic(path: Path, content: bytes) -> None:
    if path.exists():
        msg = f"Refusing to overwrite materialized file: {path}"
        raise FileExistsError(msg)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_atomic(path: Path, payload: object) -> None:
    _write_bytes_atomic(path, _canonical_json(payload) + b"\n")


def _tcga_contract_source_path() -> Path:
    module_path = tcga_data.__file__
    if module_path is None:
        msg = "Cannot attest dialect.data.tcga without a source-file path."
        raise RuntimeError(msg)
    return Path(module_path).resolve()


def _selection_policy_record() -> dict[str, object]:
    return {
        "analysis_unit": "one-participant-one-tumor-sample",
        "membership_source": "commit-matched-sequenced-case-list",
        "primary_sample_type_codes": sorted(PRIMARY_DISEASE_SAMPLE_TYPE_CODES),
        "singleton_rule": (
            "retain-sole-case-list-sample-regardless-of-sample-type"
        ),
        "repeated_participant_rule": (
            "retain-exactly-one-primary-disease-sample-otherwise-fail-closed"
        ),
        "ordering": "lexicographic-sample-barcode",
        "ordered_axis_digest": "sha256-uint64be-length-framed-utf8-v1",
    }


def _publish_claim_path(output_root: Path) -> Path:
    return output_root.with_name(
        f".{output_root.name}{_PUBLISH_CLAIM_SUFFIX}",
    )


@contextmanager
def _exclusive_publish_claim(output_root: Path) -> Iterable[None]:
    claim_path = _publish_claim_path(output_root)
    try:
        descriptor = os.open(
            claim_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
    except FileExistsError as error:
        msg = f"Another population publication holds the claim: {claim_path}"
        raise FileExistsError(msg) from error
    try:
        os.close(descriptor)
        yield
    finally:
        claim_path.unlink(missing_ok=True)


def _git_bytes(git_dir: Path, arguments: Sequence[str]) -> bytes:
    result = subprocess.run(
        ["git", f"--git-dir={git_dir}", *arguments],
        check=True,
        capture_output=True,
    )
    return result.stdout


def _validate_datahub_git_dir(git_dir: Path) -> None:
    if not git_dir.is_dir():
        msg = f"DataHub Git directory does not exist: {git_dir}"
        raise FileNotFoundError(msg)
    commit = _git_bytes(
        git_dir,
        ["rev-parse", f"{TCGA_DATAHUB_COMMIT}^{{commit}}"],
    ).decode("ascii").strip()
    tree = _git_bytes(
        git_dir,
        ["rev-parse", f"{TCGA_DATAHUB_COMMIT}^{{tree}}"],
    ).decode("ascii").strip()
    if commit != TCGA_DATAHUB_COMMIT or tree != TCGA_DATAHUB_TREE:
        msg = "DataHub Git receipt does not contain the frozen commit/tree pair."
        raise ValueError(msg)


def _validate_cohorts(cohorts: Sequence[str] | None) -> tuple[str, ...]:
    if cohorts is None:
        return TCGA_COHORTS
    selected = tuple(cohorts)
    if (
        not selected
        or len(selected) != len(set(selected))
        or any(cohort not in TCGA_COHORTS for cohort in selected)
    ):
        msg = "Cohorts must be unique exact members of the frozen TCGA family."
        raise ValueError(msg)
    selected_set = set(selected)
    return tuple(cohort for cohort in TCGA_COHORTS if cohort in selected_set)


def _materialize_cohort(
    cohort: str,
    *,
    git_dir: Path,
    staging_root: Path,
) -> dict[str, Any]:
    repository_path = tcga_datahub_case_list_path(cohort).as_posix()
    content = _git_bytes(
        git_dir,
        ["show", f"{TCGA_DATAHUB_COMMIT}:{repository_path}"],
    )
    selected = build_tcga_selected_sample_axis(content, cohort)
    receipt = TCGA_CASE_LIST_RECEIPTS[cohort]
    if _sequence_sha256(selected) != TCGA_SELECTED_SAMPLE_AXIS_SHA256[cohort]:
        msg = f"Selected sample-axis receipt changed after validation: {cohort}"
        raise RuntimeError(msg)

    cohort_dir = staging_root / cohort
    cohort_dir.mkdir()
    axis_path = cohort_dir / "sample_axis.txt"
    _write_bytes_atomic(axis_path, ("\n".join(selected) + "\n").encode("utf-8"))
    materialized_axis = axis_path.read_text(encoding="utf-8").splitlines()
    if tuple(materialized_axis) != selected:
        msg = f"Materialized sample axis failed exact readback: {cohort}"
        raise RuntimeError(msg)

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "contract": POPULATION_CONTRACT,
        "cohort": cohort,
        "source": {
            "repository": "https://github.com/cBioPortal/datahub",
            "commit": TCGA_DATAHUB_COMMIT,
            "tree": TCGA_DATAHUB_TREE,
            "repository_path": repository_path,
            "case_list_sha256": receipt.sha256,
            "case_list_bytes": len(content),
        },
        "population": {
            "source_sample_count": receipt.sample_count,
            "selected_sample_count": len(selected),
            "participant_count": receipt.participant_count,
            "removed_repeat_participant_samples": (
                receipt.sample_count - receipt.participant_count
            ),
            "ordered_sample_axis_sha256": _sequence_sha256(selected),
            "lexicographically_ordered": list(selected) == sorted(selected),
            "all_zero_rows_required_for_samples_without_retained_events": True,
        },
        "selection_policy": _selection_policy_record(),
        "contract_source": _file_record(
            _tcga_contract_source_path(),
            display_path="src/dialect/data/tcga.py",
        ),
        "outputs": {
            "sample_axis": _file_record(
                axis_path,
                display_path=f"{cohort}/sample_axis.txt",
            ),
        },
    }
    _write_json_atomic(cohort_dir / "population_manifest.json", manifest)
    return manifest


def materialize_tcga_revision_population(
    datahub_git_dir: str | Path,
    out: str | Path,
    *,
    cohorts: Sequence[str] | None = None,
) -> Path:
    """Build and atomically publish immutable TCGA revision sample axes."""
    git_dir = Path(datahub_git_dir).resolve()
    # Preserve the final path component so even a broken output symlink is refused.
    output_root = Path(os.path.abspath(out))  # noqa: PTH100
    selected_cohorts = _validate_cohorts(cohorts)
    _validate_datahub_git_dir(git_dir)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    with _exclusive_publish_claim(output_root):
        if os.path.lexists(output_root):
            msg = f"Refusing to reuse existing output root: {output_root}"
            raise FileExistsError(msg)
        staging_root = Path(
            tempfile.mkdtemp(
                prefix=f".{output_root.name}.staging-",
                dir=output_root.parent,
            ),
        )
        published = False
        try:
            manifests = [
                _materialize_cohort(
                    cohort,
                    git_dir=git_dir,
                    staging_root=staging_root,
                )
                for cohort in selected_cohorts
            ]
            cohort_records = [
                {
                    "cohort": cohort,
                    "manifest_sha256": _sha256(
                        staging_root / cohort / "population_manifest.json",
                    ),
                }
                for cohort in selected_cohorts
            ]
            root_manifest = {
                "schema_version": SCHEMA_VERSION,
                "contract": POPULATION_CONTRACT,
                "source": {
                    "repository": "https://github.com/cBioPortal/datahub",
                    "commit": TCGA_DATAHUB_COMMIT,
                    "tree": TCGA_DATAHUB_TREE,
                },
                "selection_policy": _selection_policy_record(),
                "contract_source": _file_record(
                    _tcga_contract_source_path(),
                    display_path="src/dialect/data/tcga.py",
                ),
                "cohorts": list(selected_cohorts),
                "cohort_count": len(selected_cohorts),
                "cohort_manifests": cohort_records,
                "totals": {
                    "source_sample_count": sum(
                        manifest["population"]["source_sample_count"]
                        for manifest in manifests
                    ),
                    "selected_sample_count": sum(
                        manifest["population"]["selected_sample_count"]
                        for manifest in manifests
                    ),
                    "participant_count": sum(
                        manifest["population"]["participant_count"]
                        for manifest in manifests
                    ),
                    "removed_repeat_participant_samples": sum(
                        manifest["population"][
                            "removed_repeat_participant_samples"
                        ]
                        for manifest in manifests
                    ),
                },
                "generator": _file_record(
                    Path(__file__).resolve(),
                    display_path=(
                        "analysis/materialize_tcga_revision_population.py"
                    ),
                ),
            }
            _write_json_atomic(
                staging_root / "population_manifest.json",
                root_manifest,
            )
            if os.path.lexists(output_root):
                msg = f"Output root appeared during materialization: {output_root}"
                raise FileExistsError(msg)
            staging_root.rename(output_root)
            _fsync_directory(output_root.parent)
            published = True
        finally:
            if not published and staging_root.exists():
                shutil.rmtree(staging_root)
    return output_root


def main() -> None:
    """Materialize population axes from explicit command-line paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datahub-git-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cohorts", nargs="+")
    args = parser.parse_args()
    result = materialize_tcga_revision_population(
        args.datahub_git_dir,
        args.out,
        cohorts=args.cohorts,
    )
    print(result)


if __name__ == "__main__":
    main()
