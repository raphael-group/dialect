"""Prepare the focused TCGA revision inputs without human-attestation machinery.

The scientific contract is stored in ``analysis/tcga_revision_config.json``.  This
module has two resumable stages:

``inputs``
    Restrict the pinned cBioPortal MAFs to the participant-unique sample axes and
    apply the repository's deterministic full-variant canonicalization.

``providers``
    Generate CBaSE, DIG, and native sample-specific MutSig inputs without running
    association fitting.  Existing provider-stage receipts are validated and reused.
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
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import dialect.data.tcga as tcga_data
import dialect.data.variants as variant_data
from analysis.materialize_tcga_revision_inputs import (
    _FileReceipt,
    _stream_canonicalize_maf,
)
from dialect.data.tcga import (
    TCGA_COHORTS,
    TCGA_DATAHUB_COMMIT,
    parse_tcga_sequenced_case_list,
    tcga_datahub_case_list_path,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

SCHEMA_VERSION: Final = "1.0.0"
INPUT_CONTRACT: Final = "focused-participant-axis-canonical-maf-v1"
PROVIDER_CONTRACT: Final = "focused-cbase-dig-mutsig-inputs-v1"
CONFIG_PATH: Final = Path(__file__).with_name("tcga_revision_config.json")
REQUIRED_PROVIDER_FILES: Final = (
    "bmr_pmfs.csv",
    "bmr_pmfs.dig.csv",
    "count_matrix.csv",
    "sample_axis.txt",
    "cbase_stage_receipt.tsv",
    "dig_stage_receipt.tsv",
)
REQUIRED_MUTSIG_FILES: Final = (
    "persample_genes.txt",
    "persample_lambda.f32",
    "persample_meta.txt",
    "persample_patients.txt",
    "persample_receipt.tsv",
)
THREAD_ENV: Final = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


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


def _sequence_sha256(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
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
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_atomic(path: Path, value: object) -> None:
    _write_atomic(path, _canonical_json(value) + b"\n")


def _load_config() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    analysis = config.get("analysis", {})
    execution = config.get("execution", {})
    if (
        config.get("schema_version") != SCHEMA_VERSION
        or analysis.get("cohort_count") != len(TCGA_COHORTS)
        or analysis.get("participant_count") != 10433
        or analysis.get("top_k") != 500
        or analysis.get("providers") != ["cbase", "dig", "mutsig"]
        or analysis.get("primary_provider") != "mutsig"
        or analysis.get("epsilon_prefilter") != "none"
        or analysis.get("probability_floor_or_provider_fallback") != "none"
        or execution.get("observed_data_runs") != 1
    ):
        msg = "Focused revision configuration violates the frozen contract."
        raise ValueError(msg)
    return config


def _parse_cohorts(value: str | None) -> tuple[str, ...]:
    if value is None:
        return TCGA_COHORTS
    requested = tuple(part.strip() for part in value.split(",") if part.strip())
    if not requested or len(requested) != len(set(requested)):
        msg = "--cohorts must contain unique TCGA abbreviations."
        raise ValueError(msg)
    unknown = sorted(set(requested) - set(TCGA_COHORTS))
    if unknown:
        msg = f"Unknown TCGA cohorts: {unknown}"
        raise ValueError(msg)
    selected = set(requested)
    return tuple(cohort for cohort in TCGA_COHORTS if cohort in selected)


def validate_input_root(
    input_root: Path,
    cohorts: Sequence[str],
) -> dict[str, Any]:
    """Validate canonical MAFs and sample axes against the input manifest."""
    manifest_path = input_root / "input_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = {
        str(record.get("cohort")): record
        for record in manifest.get("cohort_records", [])
    }
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("contract") != INPUT_CONTRACT
        or manifest.get("config_sha256") != _sha256(CONFIG_PATH)
        or manifest.get("cohort_count") != len(manifest.get("cohorts", []))
        or manifest.get("participant_count")
        != sum(int(record.get("sample_count", -1)) for record in records.values())
        or len(records) != len(manifest.get("cohort_records", []))
        or not set(cohorts) <= set(manifest.get("cohorts", []))
    ):
        msg = "Focused canonical input manifest is invalid."
        raise ValueError(msg)
    for cohort in cohorts:
        record = records.get(cohort, {})
        maf_path = input_root / "mafs" / f"{cohort}.maf"
        maf = record.get("canonical_maf", {})
        axis_path = input_root / "population" / cohort / "sample_axis.txt"
        axis = tuple(axis_path.read_text(encoding="utf-8").splitlines())
        if (
            maf.get("path") != f"mafs/{cohort}.maf"
            or maf.get("bytes") != maf_path.stat().st_size
            or maf.get("sha256") != _sha256(maf_path)
            or not axis
            or len(axis) != len(set(axis))
            or list(axis) != sorted(axis)
            or len(axis) != record.get("sample_count")
            or _sequence_sha256(axis) != record.get("sample_axis_sha256")
        ):
            msg = f"Focused canonical input changed: {cohort}"
            raise ValueError(msg)
    return manifest


def _git_show(git_dir: Path, object_name: str) -> bytes:
    return subprocess.run(
        ["git", f"--git-dir={git_dir}", "show", object_name],
        check=True,
        capture_output=True,
    ).stdout


def _validate_population_root(
    population_root: Path,
    cohorts: Sequence[str],
) -> dict[str, tuple[str, ...]]:
    root = json.loads(
        (population_root / "population_manifest.json").read_text(encoding="utf-8"),
    )
    if (
        root.get("contract")
        != "pinned-datahub-participant-unique-sample-axis-v1"
        or root.get("cohorts") != list(TCGA_COHORTS)
        or root.get("totals", {}).get("participant_count") != 10433
    ):
        msg = "Population root is not the corrected 10,433-participant contract."
        raise ValueError(msg)
    axes: dict[str, tuple[str, ...]] = {}
    for cohort in cohorts:
        axis_path = population_root / cohort / "sample_axis.txt"
        child_path = population_root / cohort / "population_manifest.json"
        axis = tuple(axis_path.read_text(encoding="utf-8").splitlines())
        child = json.loads(child_path.read_text(encoding="utf-8"))
        expected_hash = tcga_data.TCGA_SELECTED_SAMPLE_AXIS_SHA256[cohort]
        if (
            not axis
            or len(axis) != len(set(axis))
            or list(axis) != sorted(axis)
            or _sequence_sha256(axis) != expected_hash
            or child.get("population", {}).get("participant_count") != len(axis)
            or child.get("population", {}).get("ordered_sample_axis_sha256")
            != expected_hash
        ):
            msg = f"Population sample axis failed validation: {cohort}"
            raise ValueError(msg)
        axes[cohort] = axis
    return axes


def _materialize_cohort(  # noqa: PLR0913
    cohort: str,
    *,
    raw_maf_root: Path,
    population_root: Path,
    datahub_git_dir: Path,
    staging_root: Path,
    scratch_root: Path,
    axis: tuple[str, ...],
) -> dict[str, Any]:
    raw_path = raw_maf_root / f"{cohort}.maf"
    expected_raw_sha256 = tcga_data.TCGA_MAF_SHA256[cohort]
    raw_sha256 = _sha256(raw_path)
    if raw_sha256 != expected_raw_sha256:
        msg = f"Raw MAF differs from the pinned DataHub payload: {cohort}"
        raise ValueError(msg)
    case_path = tcga_datahub_case_list_path(cohort).as_posix()
    case_bytes = _git_show(datahub_git_dir, f"{TCGA_DATAHUB_COMMIT}:{case_path}")
    case_samples = frozenset(parse_tcga_sequenced_case_list(case_bytes, cohort))
    if not set(axis) <= case_samples:
        msg = f"Selected samples are outside the DataHub case list: {cohort}"
        raise ValueError(msg)

    population_dir = staging_root / "population" / cohort
    population_dir.mkdir(parents=True)
    axis_bytes = ("\n".join(axis) + "\n").encode("utf-8")
    _write_atomic(population_dir / "sample_axis.txt", axis_bytes)
    shutil.copyfile(
        population_root / cohort / "population_manifest.json",
        population_dir / "population_manifest.json",
    )

    canonical_path = staging_root / "mafs" / f"{cohort}.maf"
    sqlite_path = scratch_root / f"{cohort}.sqlite3"
    result = _stream_canonicalize_maf(
        raw_path,
        canonical_path,
        sqlite_path,
        raw_copy_path=None,
        expected_raw_receipt=_FileReceipt(
            bytes=raw_path.stat().st_size,
            sha256=raw_sha256,
        ),
        selected_samples=frozenset(axis),
        case_samples=case_samples,
        frozen_canonicalizer=variant_data,
    )
    for suffix in ("", "-journal", "-shm", "-wal"):
        Path(f"{sqlite_path}{suffix}").unlink(missing_ok=True)
    audit = result.audit
    return {
        "cohort": cohort,
        "sample_count": len(axis),
        "sample_axis_sha256": _sequence_sha256(axis),
        "raw_maf": {
            "bytes": raw_path.stat().st_size,
            "sha256": raw_sha256,
        },
        "canonical_maf": _file_record(canonical_path, relative_to=staging_root),
        "row_accounting": {
            "raw_rows": result.raw_rows,
            "selected_rows": result.selected_rows,
            "canonical_rows": result.output_rows,
            "removed_duplicate_rows": result.selected_rows - result.output_rows,
            "multiallelic_groups_preserved": result.multiallelic_coordinate_groups,
            "unresolved_semantic_conflicts": getattr(
                audit,
                "unresolved_semantic_conflicts",
                0,
            ),
        },
        "duplicate_resolution_policy": asdict(
            variant_data.TCGA_DUPLICATE_RESOLUTION_POLICY,
        ),
    }


def materialize_inputs(
    *,
    raw_maf_root: Path,
    population_root: Path,
    datahub_git_dir: Path,
    output_root: Path,
    cohorts: Sequence[str],
) -> Path:
    """Create a deterministic focused canonical-input root."""
    _load_config()
    axes = _validate_population_root(population_root, cohorts)
    if output_root.exists() or output_root.is_symlink():
        msg = f"Refusing to overwrite input root: {output_root}"
        raise FileExistsError(msg)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.staging-",
            dir=output_root.parent,
        ),
    )
    published = False
    try:
        (staging_root / "mafs").mkdir()
        shutil.copytree(population_root, staging_root / "population-source")
        with tempfile.TemporaryDirectory(
            prefix="dialect-canonicalize-",
            dir=output_root.parent,
        ) as scratch:
            records = [
                _materialize_cohort(
                    cohort,
                    raw_maf_root=raw_maf_root,
                    population_root=population_root,
                    datahub_git_dir=datahub_git_dir,
                    staging_root=staging_root,
                    scratch_root=Path(scratch),
                    axis=axes[cohort],
                )
                for cohort in cohorts
            ]
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "contract": INPUT_CONTRACT,
            "config": _file_record(CONFIG_PATH, relative_to=CONFIG_PATH.parent.parent),
            "config_sha256": _sha256(CONFIG_PATH),
            "population_manifest": _file_record(
                staging_root / "population-source" / "population_manifest.json",
                relative_to=staging_root,
            ),
            "datahub_commit": TCGA_DATAHUB_COMMIT,
            "cohorts": list(cohorts),
            "cohort_count": len(cohorts),
            "participant_count": sum(record["sample_count"] for record in records),
            "cohort_records": records,
        }
        _write_json_atomic(staging_root / "input_manifest.json", manifest)
        staging_root.replace(output_root)
        published = True
    finally:
        if not published and staging_root.exists():
            shutil.rmtree(staging_root)
    return output_root


def _provider_record(provider_root: Path, cohort: str) -> dict[str, Any]:
    cohort_root = provider_root / "cohorts" / cohort
    mutsig_root = provider_root / "mutsig" / cohort
    files = {
        name: _file_record(cohort_root / name, relative_to=provider_root)
        for name in REQUIRED_PROVIDER_FILES
    }
    mutsig_files = {
        name: _file_record(mutsig_root / name, relative_to=provider_root)
        for name in REQUIRED_MUTSIG_FILES
    }
    return {"cohort": cohort, "files": files, "mutsig_files": mutsig_files}


def validate_provider_root(
    provider_root: Path,
    cohorts: Sequence[str],
) -> dict[str, Any]:
    """Validate every provider artifact against the completed root manifest."""
    manifest_path = provider_root / "provider_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = {
        str(record.get("cohort")): record
        for record in manifest.get("records", [])
    }
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("contract") != PROVIDER_CONTRACT
        or manifest.get("config_sha256") != _sha256(CONFIG_PATH)
        or manifest.get("cohort_count") != len(manifest.get("cohorts", []))
        or len(records) != len(manifest.get("records", []))
        or not set(cohorts) <= set(manifest.get("cohorts", []))
    ):
        msg = "Focused provider root manifest is invalid."
        raise ValueError(msg)
    for cohort in cohorts:
        if records.get(cohort) != _provider_record(provider_root, cohort):
            msg = f"Focused provider artifacts changed: {cohort}"
            raise ValueError(msg)
    return manifest


def prepare_providers(
    *,
    input_root: Path,
    provider_root: Path,
    cohorts: Sequence[str],
    nice_increment: int,
) -> Path:
    """Generate and receipt-validate all three BMR inputs, one cohort at a time."""
    config = _load_config()
    if nice_increment != int(config["execution"]["nice_increment"]):
        msg = "Provider --nice differs from the frozen resource contract."
        raise ValueError(msg)
    validate_input_root(input_root, cohorts)
    provider_root.mkdir(parents=True, exist_ok=True)
    (provider_root / "cohorts").mkdir(exist_ok=True)
    (provider_root / "mutsig").mkdir(exist_ok=True)
    repo_root = Path(__file__).resolve().parent.parent
    for cohort in cohorts:
        environment = os.environ.copy()
        environment.update(THREAD_ENV)
        environment.update(
            {
                "PREPARE_ONLY": "1",
                "DIALECT_FOCUSED_REVISION": "1",
                "ROOT": (provider_root / "cohorts").as_posix(),
                "MAF_DIR": (input_root / "mafs").as_posix(),
                "MUTSIG_ROOT": (provider_root / "mutsig").as_posix(),
                "MUTSIG_SAMPLE_AXIS_FILE": (
                    input_root / "population" / cohort / "sample_axis.txt"
                ).as_posix(),
            },
        )
        subprocess.run(
            [
                "/usr/bin/nice",
                "-n",
                str(nice_increment),
                "bash",
                "scripts/run_cohort_pipeline.sh",
                cohort,
            ],
            check=True,
            cwd=repo_root,
            env=environment,
        )
        _provider_record(provider_root, cohort)

    records = [_provider_record(provider_root, cohort) for cohort in cohorts]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "contract": PROVIDER_CONTRACT,
        "config_sha256": _sha256(CONFIG_PATH),
        "input_manifest": _file_record(
            input_root / "input_manifest.json",
            relative_to=input_root,
        ),
        "cohorts": list(cohorts),
        "cohort_count": len(cohorts),
        "records": records,
    }
    manifest_path = provider_root / "provider_manifest.json"
    content = _canonical_json(manifest) + b"\n"
    if manifest_path.exists():
        if manifest_path.read_bytes() != content:
            msg = "Existing provider manifest differs from the completed provider tree."
            raise ValueError(msg)
    else:
        _write_atomic(manifest_path, content)
    validate_provider_root(provider_root, cohorts)
    return provider_root


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    inputs = subparsers.add_parser("inputs")
    inputs.add_argument("--raw-maf-root", type=Path, required=True)
    inputs.add_argument("--population-root", type=Path, required=True)
    inputs.add_argument("--datahub-git-dir", type=Path, required=True)
    inputs.add_argument("--output-root", type=Path, required=True)
    inputs.add_argument("--cohorts")
    providers = subparsers.add_parser("providers")
    providers.add_argument("--input-root", type=Path, required=True)
    providers.add_argument("--provider-root", type=Path, required=True)
    providers.add_argument("--cohorts")
    providers.add_argument("--nice", type=int, default=10)
    return parser


def main() -> None:
    """Run the selected focused preparation stage."""
    args = _parser().parse_args()
    cohorts = _parse_cohorts(args.cohorts)
    if args.command == "inputs":
        materialize_inputs(
            raw_maf_root=args.raw_maf_root.resolve(),
            population_root=args.population_root.resolve(),
            datahub_git_dir=args.datahub_git_dir.resolve(),
            output_root=args.output_root.absolute(),
            cohorts=cohorts,
        )
    else:
        prepare_providers(
            input_root=args.input_root.resolve(),
            provider_root=args.provider_root.absolute(),
            cohorts=cohorts,
            nice_increment=args.nice,
        )


if __name__ == "__main__":
    main()
