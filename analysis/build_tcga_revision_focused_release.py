"""Build and independently verify the focused DIALECT submission release.

The release is a deterministic, hash-manifested verification bundle. It includes
public receipts and derived results while excluding raw tumor-level inputs, sample
axes, sample-specific MutSig tensors, and restricted MutSig source code.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import inspect
import json
import os
import subprocess
import tarfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Final

from analysis import calibrate_tcga_revision_focused as calibration
from analysis import focused_revision_provenance as provenance
from analysis import freeze_tcga_revision_reporting_rule as rule_module
from analysis import postprocess_tcga_revision_focused as postprocess
from analysis import prepare_tcga_revision_focused as preparation
from analysis import report_tcga_revision_focused as reporting
from dialect.data.tcga import TCGA_COHORTS

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from typing import BinaryIO

SCHEMA_VERSION: Final = "1.0.0"
RELEASE_CONTRACT: Final = "focused-dialect-submission-release-v2"
RECEIPT_CONTRACT: Final = "focused-dialect-submission-release-receipt-v2"
SOURCE_RECORD_CONTRACT: Final = "focused-fit-release-source-boundary-v1"
MANIFEST_NAME: Final = "release_manifest.json"
SOURCE_RECORD_NAME: Final = "provenance/source_boundary.json"
FIT_ATTESTATION_MEMBER: Final = "provenance/fit_execution_attestation.json"
DOCUMENT_MANIFEST_NAME: Final = "document_manifest.json"
README_NAME: Final = "README.md"
CALIBRATION_SUMMARY_MEMBER: Final = (
    "results/calibration/calibration_summary.json"
)
REPORTING_RULE_MEMBER: Final = "results/reporting_rule.json"
REQUIRED_DOCUMENTS: Final = {
    "manuscript.tex",
    "manuscript.pdf",
    "marked_manuscript.pdf",
    "response_to_reviewers.pdf",
    "supporting_information.tex",
    "supporting_information.pdf",
    "rebuttal.md",
}
REQUIRED_REPORT_OUTPUTS: Final = {
    "figure6.pdf",
    "figure6_burden_bins.csv",
    "fit_diagnostics_summary.csv",
    "provider_overlap.csv",
    "runtime_summary.csv",
    "table_s5.csv",
    "table_s5.tex",
    "top_primary_pairs.csv",
}
RELEASED_PROVIDER_FILES: Final = (
    "bmr_pmfs.csv",
    "cbase_stage_receipt.tsv",
    "bmr_pmfs.dig.csv",
    "dig_stage_receipt.tsv",
)
_FORBIDDEN_MEMBER_BASENAMES: Final = {
    "cohort_burden_source.csv",
    "count_matrix.csv",
    "figure6_burden_source.csv",
    "sample_axis.txt",
    "persample_genes.txt",
    "persample_lambda.f32",
    "persample_meta.txt",
    "persample_patients.txt",
    "persample_receipt.tsv",
}
_JSON_LIMIT_BYTES: Final = 32 * 1024 * 1024
_CSV_HEADER_LIMIT_BYTES: Final = 1024 * 1024
_FORBIDDEN_ROW_AXIS_COLUMNS: Final = {
    "cohort_row",
    "patient",
    "patient_id",
    "sample",
    "sample_id",
    "tumor",
    "tumor_id",
}
_FIGURE6_BIN_COLUMNS: Final = {
    "cohort",
    "expected_log1p_bin_lower",
    "expected_log1p_bin_upper",
    "observed_log1p_bin_lower",
    "observed_log1p_bin_upper",
    "provider",
    "tumor_count",
}


@dataclass(frozen=True, slots=True)
class Member:
    """One immutable archive member and its source bytes."""

    name: str
    path: Path | None
    content: bytes | None
    size: int
    sha256: str


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_name(name: str) -> str:
    path = PurePosixPath(name)
    if (
        not name
        or not name.isascii()
        or path.is_absolute()
        or path.as_posix() != name
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        msg = f"Unsafe release member: {name!r}"
        raise ValueError(msg)
    return name


def _assert_public_member_name(name: str) -> None:
    path = PurePosixPath(_safe_name(name))
    lowered = name.casefold()
    if (
        path.name in _FORBIDDEN_MEMBER_BASENAMES
        or path.name.startswith("persample_")
        or path.suffix.casefold() == ".maf"
        or lowered.startswith("external/mutsig")
        or "/external/mutsig" in lowered
    ):
        msg = (
            "Release plan includes a forbidden sample-level or restricted member: "
            f"{name}"
        )
        raise ValueError(msg)


def _file_member(name: str, path: Path) -> Member:
    _assert_public_member_name(name)
    if not path.is_file() or path.is_symlink():
        msg = f"Release input is not a regular file: {path}"
        raise ValueError(msg)
    return Member(
        name=_safe_name(name),
        path=path,
        content=None,
        size=path.stat().st_size,
        sha256=_sha256_path(path),
    )


def _bytes_member(name: str, content: bytes) -> Member:
    _assert_public_member_name(name)
    return Member(
        name=_safe_name(name),
        path=None,
        content=content,
        size=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )


def _record_matches_path(record: object, path: Path, *, expected_path: str) -> bool:
    return (
        isinstance(record, dict)
        and record.get("path") == expected_path
        and record.get("bytes") == path.stat().st_size
        and record.get("sha256") == _sha256_path(path)
        and path.is_file()
        and not path.is_symlink()
    )


def _document_members(document_root: Path) -> list[Member]:
    manifest_path = document_root / DOCUMENT_MANIFEST_NAME
    if not manifest_path.is_file() or manifest_path.is_symlink():
        msg = f"Submission document root lacks {DOCUMENT_MANIFEST_NAME}."
        raise ValueError(msg)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    outputs = manifest.get("outputs", {}) if isinstance(manifest, dict) else {}
    observed = {
        path.relative_to(document_root).as_posix()
        for path in document_root.rglob("*")
        if path.is_file()
    }
    if any(path.is_symlink() for path in document_root.rglob("*")):
        msg = "Submission document root may not contain symlinks."
        raise ValueError(msg)
    if (
        not isinstance(outputs, dict)
        or set(outputs) != REQUIRED_DOCUMENTS
        or observed != {*REQUIRED_DOCUMENTS, DOCUMENT_MANIFEST_NAME}
    ):
        msg = "Submission document manifest must list the exact required files."
        raise ValueError(msg)
    for name in REQUIRED_DOCUMENTS:
        if not _record_matches_path(
            outputs[name],
            document_root / name,
            expected_path=name,
        ):
            msg = f"Submission document changed after its manifest was written: {name}"
            raise ValueError(msg)
    return [
        _file_member(f"documents/{name}", document_root / name)
        for name in sorted(REQUIRED_DOCUMENTS)
    ] + [
        _file_member(f"documents/{DOCUMENT_MANIFEST_NAME}", manifest_path),
    ]


def _call_validator(
    validator: Callable[..., object],
    *args: object,
    **available: object,
) -> object:
    """Call a validator with the subset of v1/v2 keywords that it supports."""
    signature = inspect.signature(validator)
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    selected = {
        key: value
        for key, value in available.items()
        if accepts_kwargs or key in signature.parameters
    }
    return validator(*args, **selected)


def _protocol_records() -> tuple[tuple[str, str, str | None], ...]:
    config = calibration._load_config()  # noqa: SLF001
    helper = getattr(calibration, "_protocol_cells", None)
    if helper is None:
        return tuple(
            (cohort, provider, None)
            for cohort in config["cells"]["cohorts"]
            for provider in config["cells"]["providers"]
        )
    cells = helper() if not inspect.signature(helper).parameters else helper(config)
    normalized = []
    for cell in cells:
        if isinstance(cell, (tuple, list)) and len(cell) in {2, 3}:
            role = str(cell[2]) if len(cell) == 3 else None
            normalized.append((str(cell[0]), str(cell[1]), role))
        elif isinstance(cell, dict):
            raw_role = cell.get("role")
            normalized.append(
                (
                    str(cell["cohort"]),
                    str(cell["provider"]),
                    str(raw_role) if raw_role is not None else None,
                ),
            )
        else:
            raw_role = getattr(cell, "role", None)
            normalized.append(
                (
                    str(cell.cohort),
                    str(cell.provider),
                    str(raw_role) if raw_role is not None else None,
                ),
            )
    coordinates = {(cohort, provider) for cohort, provider, _role in normalized}
    if len(normalized) != len(coordinates):
        msg = "Calibration protocol contains duplicate cells."
        raise ValueError(msg)
    return tuple(normalized)


def _protocol_cells() -> tuple[tuple[str, str], ...]:
    return tuple((cohort, provider) for cohort, provider, _role in _protocol_records())


def _validate_upstream(  # noqa: PLR0913
    *,
    repository_root: Path,
    input_root: Path,
    provider_root: Path,
    run_root: Path,
    postprocess_root: Path,
    calibration_root: Path,
    report_root: Path,
    rule_path: Path,
    fit_attestation_path: Path,
    fit_commit: str,
    release_commit: str,
    runtime_executable: Path,
) -> dict[str, Any]:
    reporting._require_reportable_rule(  # noqa: SLF001
        calibration_root=calibration_root,
        postprocess_root=postprocess_root,
        rule_path=rule_path,
        run_root=run_root,
        provider_root=provider_root,
        action="release validation",
    )
    attestation = provenance.validate_fit_attestation(
        fit_attestation_path,
        repository_root=repository_root,
        input_root=input_root,
        provider_root=provider_root,
        run_root=run_root,
        cohorts=TCGA_COHORTS,
        fit_commit=fit_commit,
        release_commit=release_commit,
        runtime_executable=runtime_executable,
    )
    _call_validator(
        postprocess.validate_derived_root,
        postprocess_root,
        TCGA_COHORTS,
        run_root=run_root,
        provider_root=provider_root,
        input_root=input_root,
        fit_attestation_path=fit_attestation_path,
        fit_attestation=fit_attestation_path,
    )
    validate_rule = getattr(rule_module, "validate_rule", None)
    if validate_rule is not None:
        _call_validator(
            validate_rule,
            rule_path,
            calibration_root=calibration_root,
            postprocess_root=postprocess_root,
            run_root=run_root,
            provider_root=provider_root,
            input_root=input_root,
            fit_attestation_path=fit_attestation_path,
            fit_attestation=fit_attestation_path,
        )
    report_manifest = _call_validator(
        reporting.validate_report,
        report_root,
        run_root=run_root,
        provider_root=provider_root,
        input_root=input_root,
        postprocess_root=postprocess_root,
        calibration_root=calibration_root,
        rule_path=rule_path,
        reporting_rule=rule_path,
        fit_attestation_path=fit_attestation_path,
        fit_attestation=fit_attestation_path,
    )
    if not isinstance(report_manifest, dict):
        msg = "Focused report validator did not return its manifest."
        raise TypeError(msg)
    rule_record = report_manifest.get("inputs", {}).get("reporting_rule", {})
    if not (
        isinstance(rule_record, dict)
        and rule_record.get("bytes") == rule_path.stat().st_size
        and rule_record.get("sha256") == _sha256_path(rule_path)
    ):
        msg = "Reporting rule differs from the rule used to build final artifacts."
        raise ValueError(msg)
    return attestation


def _result_members(  # noqa: PLR0913
    *,
    input_root: Path,
    provider_root: Path,
    run_root: Path,
    postprocess_root: Path,
    calibration_root: Path,
    report_root: Path,
    rule_path: Path,
    fit_attestation_path: Path,
) -> list[Member]:
    reporting._require_reportable_rule(  # noqa: SLF001
        calibration_root=calibration_root,
        postprocess_root=postprocess_root,
        rule_path=rule_path,
        run_root=run_root,
        provider_root=provider_root,
        action="release assembly",
    )
    members = [
        _file_member(
            "provenance/config/tcga_revision_config.json",
            preparation.CONFIG_PATH,
        ),
        _file_member(
            "provenance/config/tcga_revision_calibration_config.json",
            calibration.CONFIG_PATH,
        ),
        _file_member(
            "provenance/input/input_manifest.json",
            input_root / "input_manifest.json",
        ),
        _file_member(
            "provenance/provider/provider_manifest.json",
            provider_root / "provider_manifest.json",
        ),
        _file_member(FIT_ATTESTATION_MEMBER, fit_attestation_path),
        _file_member(
            "provenance/run/run_manifest.json",
            run_root / "run_manifest.json",
        ),
        _file_member(
            "provenance/run/completion_manifest.json",
            run_root / "completion_manifest.json",
        ),
        _file_member(
            "results/postprocess/postprocess_manifest.json",
            postprocess_root / postprocess.ROOT_MANIFEST_NAME,
        ),
        _file_member(
            "results/calibration/run_manifest.json",
            calibration_root / calibration.RUN_MANIFEST_NAME,
        ),
        _file_member(
            "results/calibration/calibration_summary.json",
            calibration_root / calibration.SUMMARY_NAME,
        ),
        _file_member(
            "results/calibration/calibration_cells.csv",
            calibration_root / calibration.SUMMARY_TABLE_NAME,
        ),
        _file_member("results/reporting_rule.json", rule_path),
    ]
    for cohort in TCGA_COHORTS:
        members.extend(
            _file_member(
                f"provenance/provider/cohorts/{cohort}/{name}",
                provider_root / "cohorts" / cohort / name,
            )
            for name in RELEASED_PROVIDER_FILES
        )
        projection = provenance.public_cohort_contract(
            run_root / "contracts" / f"{cohort}.json",
        )
        members.append(
            _bytes_member(
                f"provenance/run/contracts/{cohort}.json",
                _canonical_json(projection) + b"\n",
            ),
        )
        members.extend(
            _file_member(
                f"provenance/run/tasks/{cohort}/{provider}/task_manifest.json",
                run_root / "tasks" / cohort / provider / "task_manifest.json",
            )
            for provider in ("cbase", "dig", "mutsig")
        )
        members.extend(
            _file_member(
                f"results/postprocess/{cohort}/{name}",
                postprocess_root / cohort / name,
            )
            for name in (postprocess.RESULT_NAME, postprocess.COHORT_MANIFEST_NAME)
        )
    members.extend(
        _file_member(f"results/report/{path.name}", path)
        for path in sorted(report_root.iterdir())
        if path.is_file()
    )
    for cohort, provider in _protocol_cells():
        task_root = calibration_root / "tasks" / cohort / provider
        members.extend(
            _file_member(
                f"results/calibration/tasks/{cohort}/{provider}/{name}",
                task_root / name,
            )
            for name in (calibration.TASK_DATA_NAME, calibration.TASK_MANIFEST_NAME)
        )
    return members


def _source_record(
    *,
    repository_root: Path,
    fit_commit: str,
    release_commit: str,
    attestation: Mapping[str, Any],
) -> Member:
    observed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    source = attestation.get("source", {})
    if (
        observed != release_commit
        or status
        or source.get("fit_source_commit") != fit_commit
        or source.get("release_source_commit") != release_commit
        or source.get("fit_is_ancestor_of_release") is not True
        or source.get("raw_fit_sources_unchanged_at_release") is not True
    ):
        msg = "Release source boundary differs from its fit attestation."
        raise ValueError(msg)
    record = {
        "schema_version": SCHEMA_VERSION,
        "contract": SOURCE_RECORD_CONTRACT,
        "repository": "raphael-group/dialect",
        "fit_source_commit": fit_commit,
        "release_source_commit": release_commit,
        "fit_is_ancestor_of_release": True,
        "raw_fit_sources_unchanged_at_release": True,
        "restricted_mutsig_source_included": False,
        "raw_tumor_level_inputs_included": False,
        "sample_identifiers_included": False,
    }
    return _bytes_member(SOURCE_RECORD_NAME, _canonical_json(record) + b"\n")


def _rule_description(rule: Mapping[str, Any]) -> str:
    status = str(rule.get("inference_status", "reportable"))
    primary = rule.get("primary_q_threshold")
    adjustment = rule.get(
        "primary_adjustment",
        rule.get("multiplicity", "unknown adjustment"),
    )
    if isinstance(primary, (int, float)):
        return (
            f"Inference status is {status}; the primary rule is {adjustment} "
            f"q <= {primary:g}."
        )
    return (
        f"Inference status is {status}; see results/reporting_rule.json for the "
        "frozen rule."
    )


def _readme(
    *,
    rule: Mapping[str, Any],
    fit_commit: str,
    release_commit: str,
) -> Member:
    content = (
        "# DIALECT focused revision release\n\n"
        "This deterministic verification bundle accompanies the corrected matched "
        "K=500 analysis of 10,433 participant-unique tumors in 32 TCGA Pan-Cancer "
        "Atlas cohorts. MutSig is the primary inferential background; CBaSE is the "
        "continuity comparison and DIG is supplementary sensitivity. Provider overlap "
        "is descriptive, never a voting rule.\n\n"
        f"{_rule_description(rule)} Direction is assigned from fitted rho only after "
        "the nondirectional profile likelihood-ratio test.\n\n"
        f"Fit source commit: `{fit_commit}`. Release source commit: `{release_commit}` "
        "in `raphael-group/dialect`.\n\n"
        "## Integrity and privacy\n\n"
        "The release manifest hashes every member and the verifier checks the semantic "
        "receipt chain from input and provider manifests through raw task receipts, "
        "postprocessing, calibration, the frozen rule, reporting, and documents. "
        "Public cohort contracts preserve scientific hashes while removing host paths. "
        "The exact cohort-level CBaSE and DIG PMFs and their stage receipts are "
        "included and hash-bound to the provider manifest. The bundle excludes raw "
        "MAFs, count matrices, sample axes and identifiers, sample-specific MutSig "
        "tensors, per-tumor source tables, and restricted MutSig source. Figure 6 "
        "source data use fixed aggregate bins only.\n\n"
        "CBaSE provider generation uses stochastic initialization upstream. Its "
        "included PMFs, provider-output hashes, and completed fit receipts are "
        "therefore authoritative; this verification bundle does not claim "
        "byte-identical provider regeneration from source alone.\n\n"
        "The v1 raw task receipts did not capture source and runtime hashes at task "
        "execution. The fit attestation is therefore an honest post-run source, "
        "runtime, and receipt reconstruction against a continuously monitored clean "
        "HEAD; it is not a task-level attestation of bytes loaded in process "
        "memory.\n\n"
        "## Verify\n\n"
        "```bash\n"
        "python -m analysis.build_tcga_revision_focused_release \\\n"
        "  --verify-archive <release.tar.gz> \\\n"
        "  --receipt <release.receipt.json>\n"
        "```\n"
    ).encode()
    return _bytes_member(README_NAME, content)


def _manifest(
    members: Sequence[Member],
    *,
    fit_commit: str,
    release_commit: str,
) -> bytes:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "contract": RELEASE_CONTRACT,
        "fit_source_commit": fit_commit,
        "release_source_commit": release_commit,
        "member_count": len(members),
        "members": [
            {"path": member.name, "bytes": member.size, "sha256": member.sha256}
            for member in members
        ],
        "privacy": {
            "raw_tumor_level_inputs": "excluded",
            "sample_identifiers": "excluded",
            "sample_specific_mutsig_tensor": "excluded",
            "restricted_mutsig_source": "excluded",
        },
    }
    return _canonical_json(payload) + b"\n"


def _add_member(archive: tarfile.TarFile, member: Member) -> None:
    info = tarfile.TarInfo(member.name)
    info.size = member.size
    info.mode = 0o444
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    if member.path is not None:
        with member.path.open("rb") as handle:
            archive.addfile(info, handle)
    else:
        archive.addfile(info, BytesIO(member.content))


def _write_archive(path: Path, members: Sequence[Member], manifest: bytes) -> None:
    manifest_member = _bytes_member(MANIFEST_NAME, manifest)
    with (
        path.open("xb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
        tarfile.open(
            fileobj=compressed,
            mode="w",
            format=tarfile.USTAR_FORMAT,
        ) as archive,
    ):
        for member in sorted(
            (*members, manifest_member),
            key=lambda item: item.name,
        ):
            _add_member(archive, member)
        raw.flush()
        os.fsync(raw.fileno())


def _hash_stream(handle: BinaryIO) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    while chunk := handle.read(1024 * 1024):
        digest.update(chunk)
        size += len(chunk)
    return size, digest.hexdigest()


def _json_member(archive: tarfile.TarFile, name: str) -> dict[str, Any]:
    info = archive.getmember(name)
    if info.size > _JSON_LIMIT_BYTES:
        msg = f"Release JSON member is unexpectedly large: {name}"
        raise ValueError(msg)
    handle = archive.extractfile(info)
    if handle is None:
        msg = f"Release JSON member cannot be read: {name}"
        raise ValueError(msg)
    value = json.loads(handle.read().decode("utf-8"))
    if not isinstance(value, dict):
        msg = f"Release JSON member is not an object: {name}"
        raise TypeError(msg)
    return value


def _verified_json_member(
    archive: tarfile.TarFile,
    name: str,
    records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Read one bounded JSON member only after verifying its manifest digest."""
    info = archive.getmember(name)
    if info.size > _JSON_LIMIT_BYTES:
        msg = f"Release JSON member is unexpectedly large: {name}"
        raise ValueError(msg)
    record = records.get(name)
    if not isinstance(record, dict):
        msg = f"Release manifest omits gate evidence: {name}"
        raise TypeError(msg)
    handle = archive.extractfile(info)
    if handle is None:
        msg = f"Release JSON member cannot be read: {name}"
        raise ValueError(msg)
    content = handle.read()
    if (
        record.get("bytes") != len(content)
        or record.get("sha256") != hashlib.sha256(content).hexdigest()
    ):
        msg = f"Release member digest differs: {name}"
        raise ValueError(msg)
    value = json.loads(content.decode("utf-8"))
    if not isinstance(value, dict):
        msg = f"Release JSON member is not an object: {name}"
        raise TypeError(msg)
    return value


def _require_embedded_reportable_gate(
    archive: tarfile.TarFile,
    records: Mapping[str, Mapping[str, Any]],
) -> None:
    """Reject a withheld archive before opening any association result member."""
    summary = _verified_json_member(
        archive,
        CALIBRATION_SUMMARY_MEMBER,
        records,
    )
    summary_gate = summary.get("overall_gate_pass")
    if not isinstance(summary_gate, bool):
        msg = "Archived calibration summary lacks an exact boolean gate."
        raise TypeError(msg)
    if not summary_gate:
        msg = "Association-level archive verification is withheld: gate failed."
        raise RuntimeError(msg)

    rule = _verified_json_member(archive, REPORTING_RULE_MEMBER, records)
    rule_gate = rule.get("calibration_gate")
    if not isinstance(rule_gate, dict):
        msg = "Archived reporting rule lacks a calibration gate object."
        raise TypeError(msg)
    rule_gate_pass = rule_gate.get("overall_gate_pass")
    if not isinstance(rule_gate_pass, bool):
        msg = "Archived reporting rule lacks an exact boolean calibration gate."
        raise TypeError(msg)
    if not rule_gate_pass:
        msg = "Association-level archive verification is withheld by the rule."
        raise RuntimeError(msg)
    if rule.get("inference_status") != rule_module.REPORTABLE_STATUS:
        msg = "Association-level archive verification requires a reportable rule."
        raise RuntimeError(msg)

    calibration_binding = rule.get("calibration_summary")
    if isinstance(calibration_binding, dict):
        _require_record(
            calibration_binding,
            records,
            CALIBRATION_SUMMARY_MEMBER,
            label="gate rule to calibration",
        )
    elif (
        rule.get("calibration_summary_sha256")
        != records[CALIBRATION_SUMMARY_MEMBER].get("sha256")
    ):
        msg = "Archived reporting gate is not bound to calibration."
        raise ValueError(msg)


def _assert_aggregate_csv_header(archive: tarfile.TarFile, name: str) -> None:
    if PurePosixPath(name).suffix.casefold() != ".csv":
        return
    handle = archive.extractfile(name)
    if handle is None:
        msg = f"Release CSV member cannot be read: {name}"
        raise ValueError(msg)
    header = handle.readline(_CSV_HEADER_LIMIT_BYTES + 1)
    if len(header) > _CSV_HEADER_LIMIT_BYTES:
        msg = f"Release CSV header is unexpectedly large: {name}"
        raise ValueError(msg)
    columns = {
        column.strip().casefold()
        for column in next(csv.reader([header.decode("utf-8-sig")]), [])
    }
    forbidden = sorted(columns & _FORBIDDEN_ROW_AXIS_COLUMNS)
    if forbidden:
        msg = f"Release CSV exposes a sample-level row axis: {name}: {forbidden}"
        raise ValueError(msg)
    if (
        PurePosixPath(name).name == "figure6_burden_bins.csv"
        and columns != _FIGURE6_BIN_COLUMNS
    ):
        msg = "Figure 6 source is not an aggregate fixed-log1p-bin table."
        raise ValueError(msg)


def _record_matches_member(record: object, member: Mapping[str, Any]) -> bool:
    return (
        isinstance(record, dict)
        and record.get("bytes") == member.get("bytes")
        and record.get("sha256") == member.get("sha256")
    )


def _require_record(
    record: object,
    records: Mapping[str, Mapping[str, Any]],
    member_name: str,
    *,
    label: str,
) -> None:
    if member_name not in records or not _record_matches_member(
        record,
        records[member_name],
    ):
        msg = f"Release semantic receipt chain is broken at {label}."
        raise ValueError(msg)


def _require_record_path(
    record: object,
    records: Mapping[str, Mapping[str, Any]],
    member_name: str,
    *,
    expected_path: str,
    label: str,
) -> None:
    if not isinstance(record, dict) or record.get("path") != expected_path:
        msg = f"Release semantic receipt path is broken at {label}."
        raise ValueError(msg)
    _require_record(record, records, member_name, label=label)


def _verify_semantic_closure(
    archive: tarfile.TarFile,
    manifest: Mapping[str, Any],
    records: Mapping[str, Mapping[str, Any]],
) -> None:
    analysis_config_name = "provenance/config/tcga_revision_config.json"
    calibration_config_name = "provenance/config/tcga_revision_calibration_config.json"
    input_name = "provenance/input/input_manifest.json"
    provider_name = "provenance/provider/provider_manifest.json"
    run_name = "provenance/run/run_manifest.json"
    completion_name = "provenance/run/completion_manifest.json"
    attestation = _json_member(archive, FIT_ATTESTATION_MEMBER)
    source = _json_member(archive, SOURCE_RECORD_NAME)
    if (
        attestation.get("contract") != provenance.FIT_ATTESTATION_CONTRACT
        or manifest.get("fit_source_commit") != provenance.PRODUCTION_FIT_COMMIT
        or source.get("contract") != SOURCE_RECORD_CONTRACT
        or source.get("fit_source_commit") != manifest.get("fit_source_commit")
        or source.get("release_source_commit") != manifest.get("release_source_commit")
        or attestation.get("source", {}).get("fit_source_commit")
        != manifest.get("fit_source_commit")
        or attestation.get("source", {}).get("release_source_commit")
        != manifest.get("release_source_commit")
        or attestation.get("source", {}).get(
            "raw_fit_sources_unchanged_at_release",
        )
        is not True
        or attestation.get("source", {}).get("fit_is_ancestor_of_release") is not True
        or source.get("fit_is_ancestor_of_release") is not True
        or attestation.get("privacy")
        != {
            "raw_tumor_level_inputs_included": False,
            "sample_identifiers_included": False,
            "restricted_mutsig_source_included": False,
        }
        or source.get("raw_tumor_level_inputs_included") is not False
        or source.get("sample_identifiers_included") is not False
        or source.get("restricted_mutsig_source_included") is not False
    ):
        msg = "Release fit/release source boundary is inconsistent."
        raise ValueError(msg)

    input_manifest = _json_member(archive, input_name)
    provider = _json_member(archive, provider_name)
    run = _json_member(archive, run_name)
    completion = _json_member(archive, completion_name)
    if analysis_config_name not in records or calibration_config_name not in records:
        msg = "Release omits a frozen analysis or calibration configuration."
        raise ValueError(msg)
    analysis_config_sha256 = records[analysis_config_name]["sha256"]
    calibration_config_sha256 = records[calibration_config_name]["sha256"]
    _require_record_path(
        input_manifest.get("config"),
        records,
        analysis_config_name,
        expected_path="analysis/tcga_revision_config.json",
        label="input to analysis config",
    )
    if (
        input_manifest.get("config_sha256") != analysis_config_sha256
        or input_manifest.get("cohorts") != list(TCGA_COHORTS)
        or input_manifest.get("cohort_count") != len(TCGA_COHORTS)
        or provider.get("config_sha256") != analysis_config_sha256
        or provider.get("cohorts") != list(TCGA_COHORTS)
        or provider.get("cohort_count") != len(TCGA_COHORTS)
        or run.get("config_sha256") != analysis_config_sha256
        or run.get("cohorts") != list(TCGA_COHORTS)
        or run.get("providers") != ["cbase", "dig", "mutsig"]
        or run.get("top_k") != 500
        or completion.get("config_sha256") != analysis_config_sha256
        or completion.get("cohorts") != list(TCGA_COHORTS)
    ):
        msg = "Raw manifests are not bound to the released analysis configuration."
        raise ValueError(msg)
    _require_record_path(
        run.get("config"),
        records,
        analysis_config_name,
        expected_path="analysis/tcga_revision_config.json",
        label="run to analysis config",
    )
    _require_record(
        provider.get("input_manifest"),
        records,
        input_name,
        label="provider to input",
    )
    provider_entries = provider.get("records", [])
    provider_records = {
        str(entry.get("cohort")): entry
        for entry in provider_entries
        if isinstance(entry, dict)
    }
    if (
        not isinstance(provider_entries, list)
        or len(provider_records) != len(TCGA_COHORTS)
        or set(provider_records) != set(TCGA_COHORTS)
    ):
        msg = "Provider manifest does not cover the exact 32-cohort release grid."
        raise ValueError(msg)
    for cohort in TCGA_COHORTS:
        provider_files = provider_records[cohort].get("files", {})
        for filename in RELEASED_PROVIDER_FILES:
            provider_record = (
                provider_files.get(filename)
                if isinstance(provider_files, dict)
                else None
            )
            expected_path = f"cohorts/{cohort}/{filename}"
            member_name = f"provenance/provider/{expected_path}"
            _require_record_path(
                provider_record,
                records,
                member_name,
                expected_path=expected_path,
                label=f"provider artifact {expected_path}",
            )
    _require_record(
        run.get("provider_manifest"),
        records,
        provider_name,
        label="run to provider",
    )
    _require_record(
        completion.get("run_manifest"),
        records,
        run_name,
        label="completion to run",
    )
    expected_coordinates = {
        (cohort, provider_key)
        for cohort in TCGA_COHORTS
        for provider_key in ("cbase", "dig", "mutsig")
    }
    tasks = completion.get("tasks", [])
    coordinates = {
        (str(task.get("cohort")), str(task.get("provider")))
        for task in tasks
        if isinstance(task, dict)
    }
    if coordinates != expected_coordinates or len(tasks) != len(expected_coordinates):
        msg = "Release completion does not cover the exact 32x3 raw task grid."
        raise ValueError(msg)

    raw_tasks: dict[tuple[str, str], dict[str, Any]] = {}
    for task in tasks:
        cohort = str(task["cohort"])
        provider_key = str(task["provider"])
        task_name = f"provenance/run/tasks/{cohort}/{provider_key}/task_manifest.json"
        _require_record(
            task.get("manifest"),
            records,
            task_name,
            label=f"completion to {cohort}/{provider_key}",
        )
        raw_task = _json_member(archive, task_name)
        if (
            raw_task.get("cohort") != cohort
            or raw_task.get("provider") != provider_key
            or raw_task.get("top_k") != 500
            or raw_task.get("config_sha256") != analysis_config_sha256
            or set(raw_task.get("outputs", {}))
            != {
                "pairwise_interaction_results.csv",
                "single_gene_results.csv",
            }
        ):
            msg = f"Raw task manifest contract is invalid: {cohort}/{provider_key}"
            raise ValueError(msg)
        raw_tasks[(cohort, provider_key)] = raw_task

    raw_chain = attestation.get("raw_chain", {})
    _require_record(
        raw_chain.get("input_manifest"),
        records,
        input_name,
        label="attestation input",
    )
    _require_record(
        raw_chain.get("provider_manifest"),
        records,
        provider_name,
        label="attestation provider",
    )
    _require_record(
        raw_chain.get("run_manifest"),
        records,
        run_name,
        label="attestation run",
    )
    _require_record(
        raw_chain.get("completion_manifest"),
        records,
        completion_name,
        label="attestation completion",
    )
    attested_tasks = raw_chain.get("task_manifests", [])
    attested_task_records = {
        (str(record.get("cohort")), str(record.get("provider"))): record
        for record in attested_tasks
        if isinstance(record, dict)
    }
    if (
        not isinstance(attested_tasks, list)
        or len(attested_task_records) != len(expected_coordinates)
        or set(attested_task_records) != expected_coordinates
    ):
        msg = "Fit attestation does not cover the exact raw task grid."
        raise ValueError(msg)
    for coordinate, task_record in attested_task_records.items():
        cohort, provider_key = coordinate
        task_name = f"provenance/run/tasks/{cohort}/{provider_key}/task_manifest.json"
        _require_record(
            task_record.get("manifest"),
            records,
            task_name,
            label=f"attestation task {cohort}/{provider_key}",
        )
        if task_record.get("outputs") != raw_tasks[coordinate].get("outputs"):
            msg = f"Attested raw task outputs changed: {cohort}/{provider_key}"
            raise ValueError(msg)
    contract_evidence = {
        str(record.get("path", ""))
        .removeprefix("contracts/")
        .removesuffix(".json"): record
        for record in raw_chain.get("cohort_contracts", [])
        if isinstance(record, dict)
    }
    if len(contract_evidence) != len(TCGA_COHORTS):
        msg = "Fit attestation does not cover the exact cohort contract grid."
        raise ValueError(msg)
    for cohort in TCGA_COHORTS:
        contract_name = f"provenance/run/contracts/{cohort}.json"
        contract = _json_member(archive, contract_name)
        evidence = contract_evidence.get(cohort, {})
        source_contract = contract.get("source_contract", {})
        projection = contract.get("projection", {})
        serialized_contract = _canonical_json(contract)
        if (
            contract.get("contract") != provenance.PUBLIC_COHORT_CONTRACT
            or contract.get("cohort") != cohort
            or not isinstance(projection, dict)
            or projection.get("cohort") != cohort
            or projection.get("top_k") != 500
            or projection.get("focused_config_sha256") != analysis_config_sha256
            or source_contract.get("sha256") != evidence.get("sha256")
            or source_contract.get("bytes") != evidence.get("bytes")
            or source_contract.get("canonical_sha256")
            != evidence.get("canonical_sha256")
            or b"/Users/" in serialized_contract
            or b"/home/" in serialized_contract
            or b"TCGA-" in serialized_contract
        ):
            msg = f"Public cohort contract projection is invalid: {cohort}"
            raise ValueError(msg)
        for provider_key in ("cbase", "dig", "mutsig"):
            if raw_tasks[(cohort, provider_key)].get(
                "contract_sha256",
            ) != source_contract.get("canonical_sha256"):
                msg = (
                    "Raw task is not bound to public contract evidence: "
                    f"{cohort}/{provider_key}"
                )
                raise ValueError(msg)

    post_root_name = "results/postprocess/postprocess_manifest.json"
    post_root = _json_member(archive, post_root_name)
    _require_record(
        post_root.get("run_completion"),
        records,
        completion_name,
        label="postprocess to completion",
    )
    postprocess_records = {
        str(record.get("path")): record
        for record in post_root.get("cohort_manifests", [])
        if isinstance(record, dict)
    }
    expected_postprocess_manifests = {
        f"{cohort}/{postprocess.COHORT_MANIFEST_NAME}" for cohort in TCGA_COHORTS
    }
    if (
        post_root.get("cohorts") != list(TCGA_COHORTS)
        or post_root.get("cohort_count") != len(TCGA_COHORTS)
        or set(postprocess_records) != expected_postprocess_manifests
    ):
        msg = "Postprocess root does not cover the exact cohort manifest grid."
        raise ValueError(msg)
    for cohort in TCGA_COHORTS:
        cohort_manifest_name = (
            f"results/postprocess/{cohort}/{postprocess.COHORT_MANIFEST_NAME}"
        )
        result_name = f"results/postprocess/{cohort}/{postprocess.RESULT_NAME}"
        relative_manifest = f"{cohort}/{postprocess.COHORT_MANIFEST_NAME}"
        _require_record(
            postprocess_records[relative_manifest],
            records,
            cohort_manifest_name,
            label=f"postprocess root to cohort {cohort}",
        )
        cohort_manifest = _json_member(archive, cohort_manifest_name)
        cohort_providers = cohort_manifest.get("providers")
        if cohort_manifest.get("cohort") != cohort or cohort_providers != [
            "cbase",
            "dig",
            "mutsig",
        ]:
            msg = f"Postprocess cohort manifest contract is invalid: {cohort}"
            raise ValueError(msg)
        _require_record(
            cohort_manifest.get("output"),
            records,
            result_name,
            label=f"postprocess output {cohort}",
        )
        for provider_key in ("cbase", "dig", "mutsig"):
            raw_output = (
                raw_tasks[(cohort, provider_key)]
                .get("outputs", {})
                .get(
                    "pairwise_interaction_results.csv",
                )
            )
            source_record = cohort_manifest.get("sources", {}).get(provider_key)
            if not (
                isinstance(raw_output, dict)
                and isinstance(source_record, dict)
                and raw_output.get("bytes") == source_record.get("bytes")
                and raw_output.get("sha256") == source_record.get("sha256")
            ):
                msg = (
                    "Postprocess source is not bound to raw output: "
                    f"{cohort}/{provider_key}"
                )
                raise ValueError(msg)

    calibration_run_name = "results/calibration/run_manifest.json"
    calibration_summary_name = "results/calibration/calibration_summary.json"
    calibration_table_name = "results/calibration/calibration_cells.csv"
    calibration_run = _json_member(archive, calibration_run_name)
    calibration_summary = _json_member(archive, calibration_summary_name)
    _require_record_path(
        calibration_run.get("config"),
        records,
        calibration_config_name,
        expected_path="analysis/tcga_revision_calibration_config.json",
        label="calibration run to calibration config",
    )
    calibration_protocol = _protocol_records()
    calibration_cells = tuple(
        (cohort, provider) for cohort, provider, _role in calibration_protocol
    )
    calibration_roles = {
        (cohort, provider): role for cohort, provider, role in calibration_protocol
    }
    calibration_cohorts = {cohort for cohort, _provider in calibration_cells}
    calibration_providers = {provider for _cohort, provider in calibration_cells}
    expected_calibration_tasks = {
        f"tasks/{cohort}/{provider}/{calibration.TASK_MANIFEST_NAME}"
        for cohort, provider in calibration_cells
    }
    run_cells = calibration_run.get("cells")
    if isinstance(run_cells, list):
        run_cell_records = {
            (str(cell.get("cohort")), str(cell.get("provider"))): cell
            for cell in run_cells
            if isinstance(cell, dict)
        }
        run_protocol_matches = (
            len(run_cell_records) == len(calibration_cells)
            and set(run_cell_records) == set(calibration_cells)
            and all(
                calibration_roles[coordinate] is None
                or run_cell_records[coordinate].get("role")
                == calibration_roles[coordinate]
                for coordinate in calibration_cells
            )
        )
    else:
        run_cohorts = calibration_run.get("cohorts", [])
        run_providers = calibration_run.get("providers", [])
        run_protocol_matches = (
            isinstance(run_cohorts, list)
            and isinstance(run_providers, list)
            and set(run_cohorts) == calibration_cohorts
            and len(run_cohorts) == len(calibration_cohorts)
            and set(run_providers) == calibration_providers
            and len(run_providers) == len(calibration_providers)
        )
    if (
        not run_protocol_matches
        or calibration_summary.get("config_sha256") != calibration_config_sha256
        or calibration_summary.get("cell_count") != len(calibration_cells)
    ):
        msg = "Calibration manifests do not match the released protocol."
        raise ValueError(msg)
    _require_record(
        calibration_run.get("run_completion"),
        records,
        completion_name,
        label="calibration to completion",
    )
    _require_record(
        calibration_run.get("provider_manifest"),
        records,
        provider_name,
        label="calibration to provider",
    )
    _require_record(
        calibration_summary.get("run_manifest"),
        records,
        calibration_run_name,
        label="calibration summary to run",
    )
    _require_record(
        calibration_summary.get("table"),
        records,
        calibration_table_name,
        label="calibration summary to table",
    )
    calibration_task_names = {
        record.get("path")
        for record in calibration_summary.get("task_manifests", [])
        if isinstance(record, dict)
    }
    archived_calibration_tasks = {
        name.removeprefix("results/calibration/")
        for name in records
        if name.startswith("results/calibration/tasks/")
        and name.endswith(f"/{calibration.TASK_MANIFEST_NAME}")
    }
    if (
        calibration_task_names != archived_calibration_tasks
        or archived_calibration_tasks != expected_calibration_tasks
    ):
        msg = "Calibration summary does not cover the archived calibration cells."
        raise ValueError(msg)
    for record in calibration_summary.get("task_manifests", []):
        relative = str(record["path"])
        member_name = f"results/calibration/{relative}"
        _require_record(
            record,
            records,
            member_name,
            label=f"calibration task {relative}",
        )
        task = _json_member(archive, member_name)
        relative_parts = PurePosixPath(relative).parts
        if (
            len(relative_parts) != 4
            or task.get("cohort") != relative_parts[1]
            or task.get("provider") != relative_parts[2]
            or (
                calibration_roles[(relative_parts[1], relative_parts[2])] is not None
                and task.get("role")
                != calibration_roles[(relative_parts[1], relative_parts[2])]
            )
            or task.get("config_sha256") != calibration_config_sha256
            or task.get("run_completion_sha256") != records[completion_name]["sha256"]
        ):
            msg = f"Calibration task contract is invalid: {relative}"
            raise ValueError(msg)
        data_name = member_name.rsplit("/", 1)[0] + f"/{calibration.TASK_DATA_NAME}"
        _require_record(
            task.get("output"),
            records,
            data_name,
            label=f"calibration data {relative}",
        )

    rule_name = "results/reporting_rule.json"
    rule = _json_member(archive, rule_name)
    if (
        rule.get("analysis_config_sha256") != analysis_config_sha256
        or rule.get("calibration_config_sha256") != calibration_config_sha256
    ):
        msg = "Frozen rule is not bound to the released configurations."
        raise ValueError(msg)
    calibration_binding = rule.get("calibration_summary")
    postprocess_binding = rule.get("postprocess_manifest")
    if isinstance(calibration_binding, dict):
        _require_record(
            calibration_binding,
            records,
            calibration_summary_name,
            label="rule to calibration",
        )
    elif (
        rule.get("calibration_summary_sha256")
        != records[calibration_summary_name]["sha256"]
    ):
        msg = "Frozen rule is not bound to calibration."
        raise ValueError(msg)
    if isinstance(postprocess_binding, dict):
        _require_record(
            postprocess_binding,
            records,
            post_root_name,
            label="rule to postprocess",
        )
    elif rule.get("postprocess_manifest_sha256") != records[post_root_name]["sha256"]:
        msg = "Frozen rule is not bound to postprocess."
        raise ValueError(msg)

    report_manifest_name = "results/report/report_manifest.json"
    report = _json_member(archive, report_manifest_name)
    _require_record(
        report.get("inputs", {}).get("reporting_rule"),
        records,
        rule_name,
        label="report to rule",
    )
    report_outputs = report.get("outputs", {})
    if not isinstance(report_outputs, dict):
        msg = "Report manifest outputs are invalid."
        raise TypeError(msg)
    if set(report_outputs) != REQUIRED_REPORT_OUTPUTS:
        msg = "Report manifest is not the aggregate-only release inventory."
        raise ValueError(msg)
    archived_report_members = {
        name
        for name in records
        if name.startswith("results/report/") and name != report_manifest_name
    }
    if archived_report_members != {f"results/report/{name}" for name in report_outputs}:
        msg = "Report manifest does not cover the exact archived report inventory."
        raise ValueError(msg)
    for output_name, output_record in report_outputs.items():
        _require_record(
            output_record,
            records,
            f"results/report/{output_name}",
            label=f"report output {output_name}",
        )

    document_manifest_name = f"documents/{DOCUMENT_MANIFEST_NAME}"
    document_manifest = _json_member(archive, document_manifest_name)
    _require_record(
        document_manifest.get("inputs", {}).get("report_manifest"),
        records,
        report_manifest_name,
        label="documents to report",
    )
    archived_document_members = {
        name for name in records if name.startswith("documents/")
    }
    document_outputs = document_manifest.get("outputs", {})
    expected_document_members = {
        document_manifest_name,
        *(f"documents/{name}" for name in REQUIRED_DOCUMENTS),
    }
    if (
        set(document_outputs) != REQUIRED_DOCUMENTS
        or archived_document_members != expected_document_members
    ):
        msg = "Document manifest does not list the exact submission documents."
        raise ValueError(msg)
    for name, record in document_outputs.items():
        _require_record(
            record,
            records,
            f"documents/{name}",
            label=f"document {name}",
        )

    expected_payload = {
        README_NAME,
        SOURCE_RECORD_NAME,
        FIT_ATTESTATION_MEMBER,
        analysis_config_name,
        calibration_config_name,
        input_name,
        provider_name,
        run_name,
        completion_name,
        post_root_name,
        calibration_run_name,
        calibration_summary_name,
        calibration_table_name,
        rule_name,
        report_manifest_name,
        *expected_document_members,
        *archived_report_members,
    }
    for cohort in TCGA_COHORTS:
        expected_payload.update(
            {
                f"provenance/provider/cohorts/{cohort}/{filename}"
                for filename in RELEASED_PROVIDER_FILES
            },
        )
        expected_payload.add(f"provenance/run/contracts/{cohort}.json")
        expected_payload.add(
            f"results/postprocess/{cohort}/{postprocess.COHORT_MANIFEST_NAME}",
        )
        expected_payload.add(
            f"results/postprocess/{cohort}/{postprocess.RESULT_NAME}",
        )
        expected_payload.update(
            {
                f"provenance/run/tasks/{cohort}/{provider}/task_manifest.json"
                for provider in ("cbase", "dig", "mutsig")
            },
        )
    for relative in expected_calibration_tasks:
        task_member = f"results/calibration/{relative}"
        expected_payload.add(task_member)
        expected_payload.add(
            task_member.rsplit("/", 1)[0] + f"/{calibration.TASK_DATA_NAME}",
        )
    if set(records) != expected_payload:
        missing = sorted(expected_payload - set(records))
        unexpected = sorted(set(records) - expected_payload)
        msg = (
            "Release payload is not the exact closed public inventory: "
            f"missing={missing}, unexpected={unexpected}"
        )
        raise ValueError(msg)


def verify_archive(path: Path) -> dict[str, Any]:
    """Verify archive bytes, deterministic metadata, privacy, and receipt closure."""
    with tarfile.open(path, mode="r:gz") as archive:
        infos = archive.getmembers()
        names = [info.name for info in infos]
        if names != sorted(names) or len(names) != len(set(names)):
            msg = "Release archive member order or uniqueness is invalid."
            raise ValueError(msg)
        if any(
            not info.isfile()
            or info.mode != 0o444
            or info.mtime != 0
            or _safe_name(info.name) != info.name
            for info in infos
        ):
            msg = "Release archive contains invalid member metadata."
            raise ValueError(msg)
        for name in names:
            _assert_public_member_name(name)
        manifest = _json_member(archive, MANIFEST_NAME)
        manifest_records = manifest.get("members", [])
        records = {
            str(record.get("path")): record
            for record in manifest_records
            if isinstance(record, dict)
        }
        payload_names = set(names) - {MANIFEST_NAME}
        if (
            manifest.get("schema_version") != SCHEMA_VERSION
            or manifest.get("contract") != RELEASE_CONTRACT
            or manifest.get("member_count") != len(records)
            or len(records) != len(manifest_records)
            or set(records) != payload_names
        ):
            msg = "Release manifest does not cover the exact payload."
            raise ValueError(msg)
        _require_embedded_reportable_gate(archive, records)
        for name in sorted(payload_names):
            handle = archive.extractfile(name)
            if handle is None:
                msg = f"Release member cannot be read: {name}"
                raise ValueError(msg)
            size, digest = _hash_stream(handle)
            if size != records[name]["bytes"] or digest != records[name]["sha256"]:
                msg = f"Release member digest differs: {name}"
                raise ValueError(msg)
            _assert_aggregate_csv_header(archive, name)
        _verify_semantic_closure(archive, manifest, records)
    return manifest


def _publish_no_replace(staging: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        msg = f"Refusing to overwrite release output: {destination}"
        raise FileExistsError(msg)
    os.link(staging, destination)
    staging.unlink()


def build_release(  # noqa: PLR0913
    *,
    repository_root: Path,
    fit_commit: str,
    release_commit: str,
    runtime_executable: Path,
    input_root: Path,
    provider_root: Path,
    run_root: Path,
    postprocess_root: Path,
    calibration_root: Path,
    report_root: Path,
    rule_path: Path,
    fit_attestation_path: Path,
    document_root: Path,
    destination: Path,
    receipt_path: Path,
) -> Path:
    """Build, verify, and atomically publish one immutable release candidate."""
    if destination.exists() or receipt_path.exists():
        msg = "Release archive and receipt destinations must both be unused."
        raise FileExistsError(msg)
    attestation = _validate_upstream(
        repository_root=repository_root,
        input_root=input_root,
        provider_root=provider_root,
        run_root=run_root,
        postprocess_root=postprocess_root,
        calibration_root=calibration_root,
        report_root=report_root,
        rule_path=rule_path,
        fit_attestation_path=fit_attestation_path,
        fit_commit=fit_commit,
        release_commit=release_commit,
        runtime_executable=runtime_executable,
    )
    rule = json.loads(rule_path.read_text(encoding="utf-8"))
    members = [
        *_document_members(document_root),
        *_result_members(
            input_root=input_root,
            provider_root=provider_root,
            run_root=run_root,
            postprocess_root=postprocess_root,
            calibration_root=calibration_root,
            report_root=report_root,
            rule_path=rule_path,
            fit_attestation_path=fit_attestation_path,
        ),
        _source_record(
            repository_root=repository_root,
            fit_commit=fit_commit,
            release_commit=release_commit,
            attestation=attestation,
        ),
        _readme(rule=rule, fit_commit=fit_commit, release_commit=release_commit),
    ]
    names = [member.name for member in members]
    if MANIFEST_NAME in names or len(names) != len(set(names)):
        msg = "Release plan contains duplicate or reserved member names."
        raise ValueError(msg)
    manifest = _manifest(
        sorted(members, key=lambda item: item.name),
        fit_commit=fit_commit,
        release_commit=release_commit,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    receipt_staging = receipt_path.with_name(
        f".{receipt_path.name}.{os.getpid()}.tmp",
    )
    try:
        _write_archive(staging, members, manifest)
        verified = verify_archive(staging)
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "contract": RECEIPT_CONTRACT,
            "archive": {
                "path": destination.name,
                "bytes": staging.stat().st_size,
                "sha256": _sha256_path(staging),
            },
            "release_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
            "fit_source_commit": fit_commit,
            "release_source_commit": release_commit,
            "member_count": verified["member_count"],
        }
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        with receipt_staging.open("xb") as handle:
            handle.write(_canonical_json(receipt) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        _source_record(
            repository_root=repository_root,
            fit_commit=fit_commit,
            release_commit=release_commit,
            attestation=attestation,
        )
        _publish_no_replace(staging, destination)
        _publish_no_replace(receipt_staging, receipt_path)
    finally:
        staging.unlink(missing_ok=True)
        receipt_staging.unlink(missing_ok=True)
    verify_release(destination, receipt_path)
    return destination


def verify_release(archive_path: Path, receipt_path: Path) -> dict[str, Any]:
    """Verify the external receipt and every archived member."""
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    manifest = verify_archive(archive_path)
    manifest_sha256 = hashlib.sha256(_canonical_json(manifest) + b"\n").hexdigest()
    if (
        receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("contract") != RECEIPT_CONTRACT
        or receipt.get("archive", {}).get("path") != archive_path.name
        or receipt.get("archive", {}).get("bytes") != archive_path.stat().st_size
        or receipt.get("archive", {}).get("sha256") != _sha256_path(archive_path)
        or receipt.get("release_manifest_sha256") != manifest_sha256
        or receipt.get("fit_source_commit") != manifest.get("fit_source_commit")
        or receipt.get("release_source_commit") != manifest.get("release_source_commit")
        or receipt.get("member_count") != manifest.get("member_count")
    ):
        msg = "Release receipt does not match the verified archive."
        raise ValueError(msg)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-archive", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--fit-commit")
    parser.add_argument("--release-commit")
    parser.add_argument("--runtime-executable", type=Path)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--provider-root", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--postprocess-root", type=Path)
    parser.add_argument("--calibration-root", type=Path)
    parser.add_argument("--report-root", type=Path)
    parser.add_argument("--reporting-rule", type=Path)
    parser.add_argument("--fit-attestation", type=Path)
    parser.add_argument("--document-root", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def _required_build_args(args: argparse.Namespace) -> Mapping[str, object]:
    names = (
        "repository_root",
        "fit_commit",
        "release_commit",
        "runtime_executable",
        "input_root",
        "provider_root",
        "run_root",
        "postprocess_root",
        "calibration_root",
        "report_root",
        "reporting_rule",
        "fit_attestation",
        "document_root",
        "output",
        "receipt",
    )
    missing = [name for name in names if getattr(args, name) is None]
    if missing:
        msg = f"Release build is missing required arguments: {missing}"
        raise ValueError(msg)
    return {name: getattr(args, name) for name in names}


def main() -> None:
    """Build or independently verify the focused release."""
    args = _parser().parse_args()
    if args.verify_archive is not None:
        if args.receipt is None:
            msg = "--receipt is required with --verify-archive."
            raise ValueError(msg)
        print(verify_release(args.verify_archive.resolve(), args.receipt.resolve()))
        return
    values = _required_build_args(args)
    print(
        build_release(
            repository_root=Path(values["repository_root"]).resolve(),
            fit_commit=str(values["fit_commit"]),
            release_commit=str(values["release_commit"]),
            runtime_executable=Path(values["runtime_executable"]).resolve(),
            input_root=Path(values["input_root"]).resolve(),
            provider_root=Path(values["provider_root"]).resolve(),
            run_root=Path(values["run_root"]).resolve(),
            postprocess_root=Path(values["postprocess_root"]).resolve(),
            calibration_root=Path(values["calibration_root"]).resolve(),
            report_root=Path(values["report_root"]).resolve(),
            rule_path=Path(values["reporting_rule"]).resolve(),
            fit_attestation_path=Path(values["fit_attestation"]).resolve(),
            document_root=Path(values["document_root"]).resolve(),
            destination=Path(values["output"]).absolute(),
            receipt_path=Path(values["receipt"]).absolute(),
        ),
    )


if __name__ == "__main__":
    main()
