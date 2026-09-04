"""Build and independently verify the focused DIALECT submission release.

The release is a deterministic, hash-manifested verification bundle. It includes
public receipts and derived results while excluding raw tumor-level inputs, sample
axes, sample-specific MutSig tensors, and restricted MutSig source code.
"""

from __future__ import annotations

import argparse
import codecs
import csv
import gzip
import hashlib
import html
import inspect
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import unicodedata
import urllib.parse
import zipfile
import zlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pandas as pd

from analysis import calibrate_tcga_revision_focused as calibration
from analysis import focused_revision_provenance as provenance
from analysis import freeze_tcga_revision_reporting_rule as rule_module
from analysis import postprocess_tcga_revision_focused as postprocess
from analysis import prepare_tcga_revision_focused as preparation
from analysis import report_tcga_revision_focused as reporting
from analysis import run_tcga_revision_k500 as core
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
DOCUMENT_CONTRACT: Final = "focused-submission-document-set-v1"
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
    "cohort_burden_histogram.csv",
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
_CALIBRATION_ARRAY_LIMIT_BYTES: Final = 16 * 1024 * 1024
_INFERENCE_LIMIT_BYTES: Final = 512 * 1024 * 1024
_PDF_LIMIT_BYTES: Final = 128 * 1024 * 1024
_PDF_TOOL_OUTPUT_LIMIT_BYTES: Final = 64 * 1024 * 1024
_PDF_TOOL_TIMEOUT_SECONDS: Final = 60
_PDF_MAX_PAGES: Final = 500
_PDF_RASTER_PAGE_LIMIT_BYTES: Final = 64 * 1024 * 1024
_PDF_RASTER_TOTAL_LIMIT_BYTES: Final = 1024 * 1024 * 1024
_PDF_RASTER_DPI: Final = 200
_PDF_RASTER_MAX_DIMENSION: Final = 3300
_PRIVACY_DECODE_ROUNDS: Final = 4
_PUBLIC_TEXT_SUFFIXES: Final = frozenset({".csv", ".json", ".md", ".tex", ".tsv"})
_TCGA_SAMPLE_BARCODE: Final = re.compile(
    rb"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}",
    flags=re.IGNORECASE,
)
_TCGA_SAMPLE_BARCODE_TEXT: Final = re.compile(
    r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}",
    flags=re.IGNORECASE,
)
_SAMPLE_BARCODE_SCAN_OVERLAP: Final = 32
_FORBIDDEN_TEXT_CONTROL: Final = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_PDF_TOOLS: Final = (
    "pdfinfo",
    "pdftotext",
    "pdfdetach",
    "pdftoppm",
    "tesseract",
    "gs",
)
_PDF_OBJECT: Final = re.compile(
    rb"(?:^|[\r\n])\s*(\d+)\s+(\d+)\s+obj\b(.*?)\bendobj\b",
    flags=re.DOTALL,
)
_PDF_ANNOTATION: Final = re.compile(rb"/Type\s*/Annot\b")
_PDF_ANNOTATION_SUBTYPE: Final = re.compile(rb"/Subtype\s*/([A-Za-z0-9]+)\b")
_PDF_ANNOTATION_ARRAY: Final = re.compile(rb"/Annots\s*\[(.*?)\]", re.DOTALL)
_PDF_INDIRECT_REFERENCE: Final = re.compile(rb"(\d+)\s+(\d+)\s+R\b")
_FORBIDDEN_ROW_AXIS_COLUMNS: Final = {
    "aliquot",
    "aliquot_id",
    "barcode",
    "case",
    "case_id",
    "cohort_row",
    "patient",
    "patient_id",
    "participant",
    "participant_id",
    "sample",
    "sample_id",
    "submitter_id",
    "subject",
    "subject_id",
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
    manifest = _load_strict_json_path(
        manifest_path,
        public_name=f"documents/{DOCUMENT_MANIFEST_NAME}",
    )
    outputs = manifest.get("outputs", {}) if isinstance(manifest, dict) else {}
    observed = {
        path.relative_to(document_root).as_posix()
        for path in document_root.rglob("*")
        if path.is_file()
    }
    if any(path.is_symlink() for path in document_root.rglob("*")):
        msg = "Submission document root may not contain symlinks."
        raise ValueError(msg)
    inputs = manifest.get("inputs")
    if (
        set(manifest) != {"schema_version", "contract", "inputs", "outputs"}
        or manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("contract") != DOCUMENT_CONTRACT
        or not isinstance(inputs, dict)
        or set(inputs) != {"report_manifest"}
        or not isinstance(outputs, dict)
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


def _preflight_upstream_json(  # noqa: PLR0913
    *,
    input_root: Path,
    provider_root: Path,
    run_root: Path,
    postprocess_root: Path,
    calibration_root: Path,
    report_root: Path,
    rule_path: Path,
    fit_attestation_path: Path,
) -> None:
    """Reject ambiguous source JSON without crossing a failed result gate."""
    summary = _load_strict_json_path(
        calibration_root / calibration.SUMMARY_NAME,
    )
    if not rule_module.calibration_gate_pass(summary):
        msg = (
            "Association-level release validation is withheld: "
            "calibration gate failed."
        )
        raise RuntimeError(msg)

    rule = _load_strict_json_path(rule_path)
    preview_gate = rule.get("calibration_gate")
    if not isinstance(preview_gate, dict):
        msg = "Frozen reporting rule lacks a calibration gate object."
        raise TypeError(msg)
    preview_gate_pass = preview_gate.get("overall_gate_pass")
    if not isinstance(preview_gate_pass, bool):
        msg = "Frozen reporting rule lacks an exact boolean calibration gate."
        raise TypeError(msg)
    if not preview_gate_pass or rule.get("inference_status") != "reportable":
        msg = "Association-level release validation is withheld by the rule."
        raise RuntimeError(msg)

    paths = {
        preparation.CONFIG_PATH,
        calibration.CONFIG_PATH,
        rule_path,
        fit_attestation_path,
    }
    for root in (
        input_root,
        provider_root,
        run_root,
        postprocess_root,
        calibration_root,
        report_root,
    ):
        paths.update(root.rglob("*.json"))
    for path in sorted(paths, key=lambda value: value.as_posix()):
        _load_strict_json_path(path)


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
    _preflight_upstream_json(
        input_root=input_root,
        provider_root=provider_root,
        run_root=run_root,
        postprocess_root=postprocess_root,
        calibration_root=calibration_root,
        report_root=report_root,
        rule_path=rule_path,
        fit_attestation_path=fit_attestation_path,
    )
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
    with path.open("xb") as raw:
        with (
            gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=raw,
                mtime=0,
            ) as compressed,
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


def _hash_stream(
    handle: BinaryIO,
    *,
    public_name: str | None = None,
) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    scan = (
        public_name is not None
        and PurePosixPath(public_name).suffix.casefold() in _PUBLIC_TEXT_SUFFIXES
    )
    decoder = codecs.getincrementaldecoder("utf-8")(errors="strict") if scan else None
    overlap = ""
    compact_overlap = ""

    def inspect_text(value: str) -> None:
        nonlocal compact_overlap, overlap
        if _contains_forbidden_text_character(value):
            msg = f"Release text member contains a forbidden control: {public_name}"
            raise ValueError(msg)
        window = overlap + value
        compact_window = compact_overlap + re.sub(
            r"\s+",
            "",
            unicodedata.normalize("NFKC", value),
        )
        if (
            _contains_sample_barcode_text(window)
            or _contains_sample_barcode_text(compact_window)
        ):
            msg = f"Release member exposes a TCGA sample barcode: {public_name}"
            raise ValueError(msg)
        overlap = window[-_SAMPLE_BARCODE_SCAN_OVERLAP:]
        compact_overlap = compact_window[-_SAMPLE_BARCODE_SCAN_OVERLAP:]

    while chunk := handle.read(1024 * 1024):
        digest.update(chunk)
        size += len(chunk)
        if decoder is not None:
            try:
                inspect_text(decoder.decode(chunk, final=False))
            except UnicodeDecodeError as error:
                msg = f"Release text member is not canonical UTF-8: {public_name}"
                raise ValueError(msg) from error
    if decoder is not None:
        try:
            inspect_text(decoder.decode(b"", final=True))
        except UnicodeDecodeError as error:
            msg = f"Release text member is not canonical UTF-8: {public_name}"
            raise ValueError(msg) from error
    return size, digest.hexdigest()


def _contains_forbidden_text_character(value: str) -> bool:
    """Return whether text contains controls or invisible format characters."""
    if _FORBIDDEN_TEXT_CONTROL.search(value) is not None:
        return True
    return any(
        character not in "\t\n\r"
        and unicodedata.category(character).startswith("C")
        for character in value
    )


def _contains_sample_barcode_text(value: str) -> bool:
    def decode_once(encoded: str) -> str:
        decoded = html.unescape(encoded)
        decoded = urllib.parse.unquote(decoded, errors="strict")
        decoded = re.sub(
            r"\\(?:mbox|operatorname|mathrm|text|textrm)(?=\{)",
            "",
            decoded,
        )
        decoded = decoded.replace("{-}", "-").replace(r"\-", "-")
        decoded = decoded.replace("{", "").replace("}", "")
        decoded = "".join(
            "-"
            if unicodedata.category(character) == "Pd" or character == "\u2212"
            else character
            for character in decoded
        )
        return unicodedata.normalize("NFKC", decoded)

    projections = []
    normalized = unicodedata.normalize("NFKC", value)
    for _round in range(_PRIVACY_DECODE_ROUNDS):
        projections.append(normalized)
        try:
            decoded = decode_once(normalized)
        except UnicodeDecodeError:
            return True
        if decoded == normalized:
            break
        normalized = decoded
    else:
        try:
            if decode_once(normalized) != normalized:
                return True
        except UnicodeDecodeError:
            return True
    if not projections or projections[-1] != normalized:
        projections.append(normalized)
    return any(
        _TCGA_SAMPLE_BARCODE_TEXT.search(projection) is not None
        or _TCGA_SAMPLE_BARCODE_TEXT.search(re.sub(r"\s+", "", projection))
        is not None
        for projection in projections
    )


def _decode_public_text(content: bytes, *, name: str) -> str:
    """Decode one complete public text member and enforce its privacy grammar."""
    try:
        value = content.decode("utf-8")
    except UnicodeDecodeError as error:
        msg = f"Release text member is not canonical UTF-8: {name}"
        raise ValueError(msg) from error
    if _contains_forbidden_text_character(value):
        msg = f"Release text member contains a forbidden control: {name}"
        raise ValueError(msg)
    if _contains_sample_barcode_text(value):
        msg = f"Release member exposes a TCGA sample barcode: {name}"
        raise ValueError(msg)
    return value


def _canonical_tar_segments(
    infos: Sequence[tarfile.TarInfo],
) -> list[tuple[int, bytes | None]]:
    """Return the exact USTAR structure, with member payloads as wildcards."""
    segments: list[tuple[int, bytes | None]] = []
    offset = 0
    for info in infos:
        if (
            info.type != tarfile.REGTYPE
            or info.offset != offset
            or info.offset_data != offset + tarfile.BLOCKSIZE
            or info.pax_headers
            or info.sparse is not None
            or info.devmajor != 0
            or info.devminor != 0
        ):
            msg = f"Release archive is not canonical USTAR: {info.name}"
            raise ValueError(msg)
        try:
            header = info.tobuf(
                format=tarfile.USTAR_FORMAT,
                encoding="utf-8",
                errors="strict",
            )
        except (UnicodeError, ValueError) as error:
            msg = f"Release archive has a noncanonical USTAR header: {info.name}"
            raise ValueError(msg) from error
        if len(header) != tarfile.BLOCKSIZE:
            msg = f"Release archive has an extended USTAR header: {info.name}"
            raise ValueError(msg)
        segments.append((len(header), header))
        segments.append((info.size, None))
        padding = (-info.size) % tarfile.BLOCKSIZE
        if padding:
            segments.append((padding, bytes(padding)))
        offset = info.offset_data + info.size + padding

    # tarfile's streaming writer emits the two required zero blocks and pads the
    # stream to a full record, including one record when already block-aligned.
    final_size = (
        (offset + (2 * tarfile.BLOCKSIZE) + tarfile.RECORDSIZE - 1)
        // tarfile.RECORDSIZE
        * tarfile.RECORDSIZE
    )
    segments.append((final_size - offset, bytes(final_size - offset)))
    return segments


def _verify_canonical_archive_stream(
    path: Path,
    infos: Sequence[tarfile.TarInfo],
) -> None:
    """Reject hidden gzip members, trailers, and noncanonical tar bytes."""
    segments = _canonical_tar_segments(infos)
    segment_index = 0
    segment_offset = 0
    decoded_size = 0
    barcode_overlap = b""
    decoder = zlib.decompressobj(wbits=16 + zlib.MAX_WBITS)

    def inspect_decoded(chunk: bytes) -> None:
        nonlocal barcode_overlap, decoded_size, segment_index, segment_offset
        window = barcode_overlap + chunk
        if _TCGA_SAMPLE_BARCODE.search(window) is not None:
            msg = "Release archive exposes a TCGA sample barcode."
            raise ValueError(msg)
        barcode_overlap = window[-_SAMPLE_BARCODE_SCAN_OVERLAP:]
        decoded_size += len(chunk)
        chunk_offset = 0
        while chunk_offset < len(chunk):
            if segment_index >= len(segments):
                msg = "Release tar stream has trailing decoded bytes."
                raise ValueError(msg)
            segment_size, expected = segments[segment_index]
            take = min(
                len(chunk) - chunk_offset,
                segment_size - segment_offset,
            )
            if (
                expected is not None
                and chunk[chunk_offset : chunk_offset + take]
                != expected[segment_offset : segment_offset + take]
            ):
                msg = "Release tar stream is not canonical USTAR."
                raise ValueError(msg)
            chunk_offset += take
            segment_offset += take
            if segment_offset == segment_size:
                segment_index += 1
                segment_offset = 0

    with path.open("rb") as compressed:
        while chunk := compressed.read(1024 * 1024):
            if decoder.eof:
                msg = "Release gzip stream has trailing bytes or members."
                raise ValueError(msg)
            try:
                inspect_decoded(decoder.decompress(chunk))
            except zlib.error as error:
                msg = "Release gzip stream cannot be decoded."
                raise ValueError(msg) from error
            if decoder.unused_data or decoder.unconsumed_tail:
                msg = "Release gzip stream has trailing bytes or members."
                raise ValueError(msg)
    if not decoder.eof:
        msg = "Release gzip stream is truncated."
        raise ValueError(msg)
    inspect_decoded(decoder.flush())
    if (
        decoded_size != sum(size for size, _expected in segments)
        or segment_index != len(segments)
        or segment_offset != 0
    ):
        msg = "Release tar stream has a noncanonical length."
        raise ValueError(msg)


def _run_pdf_tool(
    tool: str,
    arguments: Sequence[str],
    *,
    path: Path,
    public_name: str,
    trailing_arguments: Sequence[str] = (),
) -> str:
    """Run one bounded Poppler inspection and return privacy-checked text."""
    executable = shutil.which(tool)
    if executable is None:
        msg = f"Required PDF privacy tool is unavailable: {tool}"
        raise RuntimeError(msg)
    environment = os.environ.copy()
    environment.update({"LANG": "C", "LC_ALL": "C", "TZ": "UTC"})
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        try:
            completed = subprocess.run(
                [executable, *arguments, str(path), *trailing_arguments],
                check=False,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                timeout=_PDF_TOOL_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as error:
            msg = f"PDF privacy inspection timed out: {public_name} ({tool})"
            raise ValueError(msg) from error
        outputs: list[bytes] = []
        for handle in (stdout, stderr):
            if handle.tell() > _PDF_TOOL_OUTPUT_LIMIT_BYTES:
                msg = f"PDF privacy inspection output is too large: {public_name}"
                raise ValueError(msg)
            handle.seek(0)
            outputs.append(handle.read())
    for stream_name, content in zip(("stdout", "stderr"), outputs, strict=True):
        _decode_public_text(
            content,
            name=f"{public_name} ({tool} {stream_name})",
        )
    if completed.returncode != 0:
        msg = f"PDF privacy inspection failed: {public_name} ({tool})"
        raise ValueError(msg)
    return outputs[0].decode("utf-8")


def _pdf_literal_strings(content: bytes) -> list[bytes]:
    """Decode PDF literal-string escapes from a normalized PDF."""
    strings: list[bytes] = []
    index = 0
    while index < len(content):
        if content[index] != ord("("):
            index += 1
            continue
        index += 1
        depth = 1
        value = bytearray()
        while index < len(content) and depth:
            character = content[index]
            index += 1
            if character == ord("\\"):
                if index >= len(content):
                    break
                escaped = content[index]
                index += 1
                if ord("0") <= escaped <= ord("7"):
                    digits = bytearray([escaped])
                    while (
                        len(digits) < 3
                        and index < len(content)
                        and ord("0") <= content[index] <= ord("7")
                    ):
                        digits.append(content[index])
                        index += 1
                    value.append(int(digits, 8))
                elif escaped in b"nrtbf":
                    value.append(
                        {
                            ord("n"): ord("\n"),
                            ord("r"): ord("\r"),
                            ord("t"): ord("\t"),
                            ord("b"): ord("\b"),
                            ord("f"): ord("\f"),
                        }[escaped],
                    )
                elif escaped == ord("\r"):
                    if index < len(content) and content[index] == ord("\n"):
                        index += 1
                elif escaped != ord("\n"):
                    value.append(escaped)
            elif character == ord("("):
                depth += 1
                value.append(character)
            elif character == ord(")"):
                depth -= 1
                if depth:
                    value.append(character)
            else:
                value.append(character)
        if depth:
            msg = "Ghostscript produced an unterminated PDF literal string."
            raise ValueError(msg)
        strings.append(bytes(value))
    return strings


def _pdf_hex_strings(content: bytes) -> list[bytes]:
    """Decode valid PDF hexadecimal strings from a normalized PDF."""
    strings: list[bytes] = []
    index = 0
    while index < len(content):
        if content[index : index + 1] != b"<" or content[index : index + 2] == b"<<":
            index += 1
            continue
        end = content.find(b">", index + 1)
        if end < 0:
            break
        encoded = re.sub(rb"\s+", b"", content[index + 1 : end])
        if re.fullmatch(rb"[0-9A-Fa-f]*", encoded) is not None:
            if len(encoded) % 2:
                encoded += b"0"
            strings.append(bytes.fromhex(encoded.decode("ascii")))
        index = end + 1
    return strings


def _pdf_text_projections(value: bytes) -> tuple[str, ...]:
    """Return conservative text interpretations of one PDF string."""
    projections = [value.decode("latin-1")]
    for encoding in ("utf-8", "utf-16", "utf-16-be", "utf-16-le"):
        try:
            decoded = value.decode(encoding)
        except UnicodeError:
            continue
        if decoded not in projections:
            projections.append(decoded)
    return tuple(projections)


def _scan_normalized_pdf_privacy(content: bytes, *, name: str) -> None:
    """Scan normalized PDF strings and reject non-link annotations."""
    raw_projection = content.decode("latin-1")
    if _contains_sample_barcode_text(raw_projection):
        msg = f"Release PDF exposes a TCGA sample barcode: {name}"
        raise ValueError(msg)

    objects = {
        (int(match.group(1)), int(match.group(2))): match.group(3)
        for match in _PDF_OBJECT.finditer(content)
    }
    dictionary_regions = [body.partition(b"stream")[0] for body in objects.values()]
    projections = [
        projection
        for region in dictionary_regions
        for value in (*_pdf_literal_strings(region), *_pdf_hex_strings(region))
        for projection in _pdf_text_projections(value)
    ]
    if any(_contains_sample_barcode_text(value) for value in projections) or (
        projections and _contains_sample_barcode_text("".join(projections))
    ):
        msg = f"Release PDF exposes a TCGA sample barcode: {name}"
        raise ValueError(msg)

    referenced_annotations: set[tuple[int, int]] = set()
    annotation_arrays = list(_PDF_ANNOTATION_ARRAY.finditer(content))
    if b"/Annots" in content and not annotation_arrays:
        msg = f"Release PDF has a noncanonical annotation array: {name}"
        raise ValueError(msg)
    for array in annotation_arrays:
        references = {
            (int(reference.group(1)), int(reference.group(2)))
            for reference in _PDF_INDIRECT_REFERENCE.finditer(array.group(1))
        }
        if not references:
            msg = f"Release PDF has an unsupported inline annotation: {name}"
            raise ValueError(msg)
        referenced_annotations.update(references)

    typed_annotations = {
        reference
        for reference, body in objects.items()
        if _PDF_ANNOTATION.search(body) is not None
    }
    for reference in referenced_annotations | typed_annotations:
        body = objects.get(reference)
        if body is None:
            msg = f"Release PDF has an unresolved annotation: {name}"
            raise ValueError(msg)
        subtype = _PDF_ANNOTATION_SUBTYPE.findall(body)
        if subtype != [b"Link"]:
            msg = f"Release PDF contains a non-link annotation: {name}"
            raise ValueError(msg)


def _scan_pdf_raster_privacy(
    path: Path,
    *,
    name: str,
    page_count: int,
    directory: Path,
) -> None:
    """OCR a bounded raster of every page to inspect image-only text."""
    if page_count > _PDF_MAX_PAGES:
        msg = f"Release PDF has too many pages for privacy inspection: {name}"
        raise ValueError(msg)
    total_bytes = 0
    for page in range(1, page_count + 1):
        prefix = directory / f"page-{page:04d}"
        _run_pdf_tool(
            "pdftoppm",
            (
                "-f",
                str(page),
                "-l",
                str(page),
                "-singlefile",
                "-r",
                str(_PDF_RASTER_DPI),
                "-scale-to",
                str(_PDF_RASTER_MAX_DIMENSION),
                "-png",
            ),
            path=path,
            public_name=name,
            trailing_arguments=(str(prefix),),
        )
        raster = prefix.with_suffix(".png")
        if not raster.is_file() or raster.is_symlink():
            msg = f"PDF rasterization produced an unsafe output: {name}"
            raise ValueError(msg)
        page_bytes = raster.stat().st_size
        total_bytes += page_bytes
        if (
            page_bytes > _PDF_RASTER_PAGE_LIMIT_BYTES
            or total_bytes > _PDF_RASTER_TOTAL_LIMIT_BYTES
        ):
            msg = f"PDF rasterization exceeded the privacy bound: {name}"
            raise ValueError(msg)
        _run_pdf_tool(
            "tesseract",
            (),
            path=raster,
            public_name=name,
            trailing_arguments=("stdout", "--psm", "11", "-l", "eng"),
        )
        raster.unlink()


def _scan_pdf_privacy(content: bytes, *, name: str) -> None:
    """Reject malformed, encrypted, active, or identifier-bearing PDFs."""
    if len(content) > _PDF_LIMIT_BYTES:
        msg = f"Release PDF member is unexpectedly large: {name}"
        raise ValueError(msg)
    with tempfile.TemporaryDirectory(prefix=".dialect-release-pdf-") as directory:
        path = Path(directory) / "document.pdf"
        path.write_bytes(content)
        standard = _run_pdf_tool(
            "pdfinfo",
            ("-enc", "UTF-8"),
            path=path,
            public_name=name,
        )
        fields = {
            key.strip(): value.strip()
            for line in standard.splitlines()
            for key, separator, value in (line.partition(":"),)
            if separator
        }
        try:
            page_count = int(fields.get("Pages", ""))
        except ValueError as error:
            msg = f"Release PDF lacks a valid page count: {name}"
            raise ValueError(msg) from error
        if page_count < 1 or fields.get("Encrypted") != "no":
            msg = f"Release PDF is empty or encrypted: {name}"
            raise ValueError(msg)

        for arguments in (
            ("-custom", "-enc", "UTF-8"),
            ("-meta", "-enc", "UTF-8"),
            ("-url", "-enc", "UTF-8"),
        ):
            _run_pdf_tool(
                "pdfinfo",
                arguments,
                path=path,
                public_name=name,
            )
        javascript = _run_pdf_tool(
            "pdfinfo",
            ("-js", "-enc", "UTF-8"),
            path=path,
            public_name=name,
        )
        if javascript.strip():
            msg = f"Release PDF contains active JavaScript: {name}"
            raise ValueError(msg)
        attachments = _run_pdf_tool(
            "pdfdetach",
            ("-list", "-enc", "UTF-8"),
            path=path,
            public_name=name,
        )
        if re.fullmatch(r"\s*0 embedded files\s*", attachments) is None:
            msg = f"Release PDF contains embedded files: {name}"
            raise ValueError(msg)
        _run_pdf_tool(
            "pdftotext",
            ("-enc", "UTF-8", "-nopgbrk"),
            path=path,
            public_name=name,
            trailing_arguments=("-",),
        )
        normalized_path = Path(directory) / "normalized.pdf"
        _run_pdf_tool(
            "gs",
            (
                "-q",
                "-dSAFER",
                "-dBATCH",
                "-dNOPAUSE",
                "-dPreserveAnnots=true",
                "-dCompressStreams=false",
                "-dCompressFonts=false",
                "-dCompatibilityLevel=1.4",
                "-sDEVICE=pdfwrite",
                f"-sOutputFile={normalized_path}",
            ),
            path=path,
            public_name=name,
        )
        if (
            not normalized_path.is_file()
            or normalized_path.is_symlink()
            or normalized_path.stat().st_size > _PDF_LIMIT_BYTES
        ):
            msg = f"PDF normalization produced an unsafe output: {name}"
            raise ValueError(msg)
        _scan_normalized_pdf_privacy(normalized_path.read_bytes(), name=name)
        _scan_pdf_raster_privacy(
            path,
            name=name,
            page_count=page_count,
            directory=Path(directory),
        )




def _contains_sample_barcode(value: object) -> bool:
    """Return whether a decoded JSON value contains a TCGA sample barcode."""
    if isinstance(value, dict):
        return any(
            _contains_sample_barcode(key)
            or _contains_sample_barcode(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_contains_sample_barcode(child) for child in value)
    return (
        isinstance(value, str)
        and (
            _contains_forbidden_text_character(value)
            or _contains_sample_barcode_text(value)
        )
    )


def _json_object_without_duplicates(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            msg = f"Release JSON contains a duplicate key: {key}"
            raise ValueError(msg)
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    msg = f"Release JSON contains a nonstandard numeric constant: {value}"
    raise ValueError(msg)


def _decode_strict_json_object(
    content: bytes,
    *,
    name: str,
    public: bool,
) -> dict[str, Any]:
    """Decode one bounded UTF-8 JSON object without ambiguous extensions."""
    if len(content) > _JSON_LIMIT_BYTES:
        msg = f"JSON input is unexpectedly large: {name}"
        raise ValueError(msg)
    try:
        text = (
            _decode_public_text(content, name=name)
            if public
            else content.decode("utf-8", errors="strict")
        )
    except UnicodeDecodeError as error:
        msg = f"JSON input is not canonical UTF-8: {name}"
        raise ValueError(msg) from error
    value = json.loads(
        text,
        object_pairs_hook=_json_object_without_duplicates,
        parse_constant=_reject_json_constant,
    )
    if not isinstance(value, dict):
        msg = f"JSON input is not an object: {name}"
        raise TypeError(msg)
    if public and _contains_sample_barcode(value):
        msg = f"Release member exposes a TCGA sample barcode: {name}"
        raise ValueError(msg)
    return value


def _decode_json_object(content: bytes, *, name: str) -> dict[str, Any]:
    return _decode_strict_json_object(content, name=name, public=True)


def _load_strict_json_path(
    path: Path,
    *,
    public_name: str | None = None,
) -> dict[str, Any]:
    """Strictly decode one safe source-side or external JSON object."""
    if path.is_symlink():
        msg = f"Required JSON input is unsafe: {path}"
        raise ValueError(msg)
    if not path.is_file():
        msg = f"Required JSON input is missing: {path}"
        raise FileNotFoundError(msg)
    return _decode_strict_json_object(
        path.read_bytes(),
        name=public_name or path.name,
        public=public_name is not None,
    )


def _json_member(archive: tarfile.TarFile, name: str) -> dict[str, Any]:
    info = archive.getmember(name)
    if info.size > _JSON_LIMIT_BYTES:
        msg = f"Release JSON member is unexpectedly large: {name}"
        raise ValueError(msg)
    handle = archive.extractfile(info)
    if handle is None:
        msg = f"Release JSON member cannot be read: {name}"
        raise ValueError(msg)
    return _decode_json_object(handle.read(), name=name)


def _verified_member_bytes(
    archive: tarfile.TarFile,
    name: str,
    records: Mapping[str, Mapping[str, Any]],
    *,
    limit: int,
) -> bytes:
    """Read one bounded archive member after validating its manifest record."""
    info = archive.getmember(name)
    if info.size > limit:
        msg = f"Release member is unexpectedly large: {name}"
        raise ValueError(msg)
    record = records.get(name)
    if not isinstance(record, dict):
        msg = f"Release manifest omits required evidence: {name}"
        raise TypeError(msg)
    handle = archive.extractfile(info)
    if handle is None:
        msg = f"Release member cannot be read: {name}"
        raise ValueError(msg)
    content = handle.read(limit + 1)
    if len(content) > limit:
        msg = f"Release member is unexpectedly large: {name}"
        raise ValueError(msg)
    if (
        record.get("bytes") != len(content)
        or record.get("sha256") != hashlib.sha256(content).hexdigest()
    ):
        msg = f"Release member digest differs: {name}"
        raise ValueError(msg)
    return content


def _verified_json_member(
    archive: tarfile.TarFile,
    name: str,
    records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Read one bounded JSON member only after verifying its manifest digest."""
    content = _verified_member_bytes(
        archive,
        name,
        records,
        limit=_JSON_LIMIT_BYTES,
    )
    return _decode_json_object(content, name=name)


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


def _require_exact_keys(
    value: object,
    expected: set[str],
    *,
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        msg = f"{label} violates its exact schema."
        raise ValueError(msg)
    return value


def _archive_record(
    records: Mapping[str, Mapping[str, Any]],
    member_name: str,
    *,
    path: str,
) -> dict[str, int | str]:
    record = records.get(member_name)
    if not isinstance(record, dict):
        msg = f"Release manifest omits required evidence: {member_name}"
        raise TypeError(msg)
    return {
        "path": path,
        "bytes": int(record["bytes"]),
        "sha256": str(record["sha256"]),
    }


def _calibration_arrays(
    content: bytes,
    *,
    name: str,
) -> dict[str, np.ndarray]:
    """Load one bounded NPZ without accepting extra or oversized members."""
    expected_array_names = (
        "marginal_lrt",
        "marginal_reportable",
        "scalar_fallback",
        "sentinel_pairs",
    )
    expected_members = tuple(
        f"{array_name}.npy" for array_name in expected_array_names
    )
    expected_member_set = {
        "marginal_lrt.npy",
        "marginal_reportable.npy",
        "scalar_fallback.npy",
        "sentinel_pairs.npy",
    }
    try:
        with zipfile.ZipFile(BytesIO(content)) as bundle:
            infos = bundle.infolist()
    except (OSError, zipfile.BadZipFile) as error:
        msg = f"Archived calibration NPZ cannot be decoded: {name}"
        raise ValueError(msg) from error
    if (
        tuple(info.filename for info in infos) != expected_members
        or {info.filename for info in infos} != expected_member_set
        or any(
            info.is_dir()
            or info.flag_bits & 0x1
            or info.file_size > _CALIBRATION_ARRAY_LIMIT_BYTES
            for info in infos
        )
        or sum(info.file_size for info in infos) > _CALIBRATION_ARRAY_LIMIT_BYTES
    ):
        msg = f"Archived calibration NPZ has an unsafe inventory: {name}"
        raise ValueError(msg)
    try:
        with np.load(BytesIO(content), allow_pickle=False) as bundle:
            arrays = {
                array_name: bundle[array_name].copy()
                for array_name in expected_array_names
            }
    except (OSError, ValueError) as error:
        msg = f"Archived calibration NPZ cannot be decoded: {name}"
        raise ValueError(msg) from error
    canonical = BytesIO()
    np.savez_compressed(canonical, **arrays)
    if canonical.getvalue() != content:
        msg = f"Archived calibration NPZ is not canonical: {name}"
        raise ValueError(msg)
    return arrays


def _public_calibration_contract(
    archive: tarfile.TarFile,
    records: Mapping[str, Mapping[str, Any]],
    *,
    cohort: str,
    analysis_config_sha256: str,
) -> tuple[tuple[str, ...], int]:
    name = f"provenance/run/contracts/{cohort}.json"
    contract = _verified_json_member(archive, name, records)
    _require_exact_keys(
        contract,
        {"schema_version", "contract", "cohort", "source_contract", "projection"},
        label=f"Public cohort contract {cohort}",
    )
    projection = contract.get("projection")
    if not isinstance(projection, dict):
        msg = f"Public cohort contract lacks a projection: {cohort}"
        raise TypeError(msg)
    features = projection.get("features")
    samples = projection.get("samples")
    expected_pair_policy = (
        {
            "epsilon_pretest_filter": core.TESTED_FAMILY_NO_PRETEST_FILTER,
            "marginal_effect_pretest_filter": core.TESTED_FAMILY_NO_PRETEST_FILTER,
            "pair_construction": core.TESTED_FAMILY_PAIR_CONSTRUCTION,
            "same_base_missense_nonsense": core.TESTED_FAMILY_SAME_BASE_POLICY,
            **core._pair_contract(features),  # noqa: SLF001
        }
        if isinstance(features, list)
        else None
    )
    if (
        contract.get("schema_version") != provenance.SCHEMA_VERSION
        or contract.get("contract") != provenance.PUBLIC_COHORT_CONTRACT
        or contract.get("cohort") != cohort
        or projection.get("cohort") != cohort
        or projection.get("top_k") != 500
        or projection.get("focused_config_sha256") != analysis_config_sha256
        or not isinstance(features, list)
        or len(features) != 500
        or not all(isinstance(feature, str) and feature for feature in features)
        or len(set(features)) != len(features)
        or projection.get("pair_policy") != expected_pair_policy
        or not isinstance(samples, dict)
        or not isinstance(samples.get("count"), int)
        or isinstance(samples.get("count"), bool)
        or int(samples["count"]) < 1
    ):
        msg = f"Public cohort contract cannot reproduce the pair universe: {cohort}"
        raise ValueError(msg)
    return tuple(features), int(samples["count"])


def _recompute_archived_calibration_gate(
    archive: tarfile.TarFile,
    records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Recompute every calibration endpoint before association data are opened."""
    config_name = "provenance/config/tcga_revision_calibration_config.json"
    config_bytes = _verified_member_bytes(
        archive,
        config_name,
        records,
        limit=_JSON_LIMIT_BYTES,
    )
    if config_bytes != calibration.CONFIG_PATH.read_bytes():
        msg = "Archived calibration configuration differs from the frozen verifier."
        raise ValueError(msg)
    config = calibration._load_config()  # noqa: SLF001
    analysis_config_name = "provenance/config/tcga_revision_config.json"
    analysis_config_bytes = _verified_member_bytes(
        archive,
        analysis_config_name,
        records,
        limit=_JSON_LIMIT_BYTES,
    )
    if analysis_config_bytes != preparation.CONFIG_PATH.read_bytes():
        msg = "Archived analysis configuration differs from the frozen verifier."
        raise ValueError(msg)
    completion_name = "provenance/run/completion_manifest.json"
    provider_name = "provenance/provider/provider_manifest.json"
    run_name = "results/calibration/run_manifest.json"
    table_name = "results/calibration/calibration_cells.csv"
    summary_name = CALIBRATION_SUMMARY_MEMBER
    analysis_config_sha256 = hashlib.sha256(analysis_config_bytes).hexdigest()
    completion_sha256 = str(records[completion_name]["sha256"])
    provider_sha256 = str(records[provider_name]["sha256"])
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    protocol = calibration._protocol_cells(config)  # noqa: SLF001

    run = _verified_json_member(archive, run_name, records)
    expected_run = {
        "schema_version": calibration.SCHEMA_VERSION,
        "contract": calibration.RUN_CONTRACT,
        "config": _archive_record(
            records,
            config_name,
            path="analysis/tcga_revision_calibration_config.json",
        ),
        "run_completion": _archive_record(
            records,
            completion_name,
            path="completion_manifest.json",
        ),
        "provider_manifest": _archive_record(
            records,
            provider_name,
            path="provider_manifest.json",
        ),
        "cells": [
            {"cohort": cell.cohort, "provider": cell.provider, "role": cell.role}
            for cell in protocol
        ],
        "resource_contract": dict(config["resources"]),
        "runtime_resource_observation": calibration._frozen_resource_observation(  # noqa: SLF001
            config,
        ),
        "thread_environment": dict(calibration.THREAD_ENV),
        "result_blindness": calibration._result_blindness_receipt(),  # noqa: SLF001
    }
    if run != expected_run:
        msg = "Archived calibration run manifest differs from the frozen protocol."
        raise ValueError(msg)

    contract_axes = {
        cohort: _public_calibration_contract(
            archive,
            records,
            cohort=cohort,
            analysis_config_sha256=analysis_config_sha256,
        )
        for cohort in TCGA_COHORTS
    }
    marginal = config["marginal_lrt"]
    replicates = int(marginal["replicates_per_cell"])
    pair_count = int(marginal["sentinel_pair_count"])
    rows: list[dict[str, Any]] = []
    summary_tasks: list[dict[str, Any]] = []
    nonreportable_fit_count = 0
    for cell in protocol:
        cohort = cell.cohort
        provider_key = cell.provider
        root = f"results/calibration/tasks/{cohort}/{provider_key}"
        task_name = f"{root}/{calibration.TASK_MANIFEST_NAME}"
        data_name = f"{root}/{calibration.TASK_DATA_NAME}"
        raw_task_name = (
            f"provenance/run/tasks/{cohort}/{provider_key}/task_manifest.json"
        )
        task = _verified_json_member(archive, task_name, records)
        raw_task = _verified_json_member(archive, raw_task_name, records)
        raw_outputs = raw_task.get("outputs")
        if not isinstance(raw_outputs, dict):
            msg = f"Raw task outputs are invalid: {cohort}/{provider_key}"
            raise TypeError(msg)
        single = raw_outputs.get("single_gene_results.csv")
        if not isinstance(single, dict):
            msg = f"Raw single-gene receipt is invalid: {cohort}/{provider_key}"
            raise TypeError(msg)
        features, sample_count = contract_axes[cohort]
        data = _verified_member_bytes(
            archive,
            data_name,
            records,
            limit=_CALIBRATION_ARRAY_LIMIT_BYTES,
        )
        arrays = _calibration_arrays(data, name=data_name)
        expected_pairs = calibration._sentinel_pairs(features, pair_count)  # noqa: SLF001
        likelihood_ratios = arrays.get("marginal_lrt")
        reportable = arrays.get("marginal_reportable")
        fallback = arrays.get("scalar_fallback")
        sentinel_pairs = arrays.get("sentinel_pairs")
        if (
            set(arrays)
            != {
                "marginal_lrt",
                "marginal_reportable",
                "scalar_fallback",
                "sentinel_pairs",
            }
            or not isinstance(likelihood_ratios, np.ndarray)
            or likelihood_ratios.dtype != np.dtype(np.float64)
            or likelihood_ratios.shape != (replicates, pair_count)
            or not np.isfinite(likelihood_ratios).all()
            or (likelihood_ratios < 0).any()
            or not isinstance(reportable, np.ndarray)
            or reportable.dtype != np.dtype(bool)
            or reportable.shape != (replicates, pair_count)
            or not isinstance(fallback, np.ndarray)
            or fallback.dtype != np.dtype(bool)
            or fallback.shape != (replicates, pair_count)
            or not isinstance(sentinel_pairs, np.ndarray)
            or sentinel_pairs.dtype != np.dtype(np.int32)
            or not np.array_equal(sentinel_pairs, expected_pairs)
        ):
            msg = (
                "Archived calibration arrays violate their contract: "
                f"{cohort}/{provider_key}"
            )
            raise ValueError(msg)
        fallback_count = int(fallback.sum())
        source_manifest = _archive_record(
            records,
            raw_task_name,
            path=f"tasks/{cohort}/{provider_key}/task_manifest.json",
        )
        single_gene_input = {
            "path": f"tasks/{cohort}/{provider_key}/single_gene_results.csv",
            "bytes": single.get("bytes"),
            "sha256": single.get("sha256"),
        }
        fit_kernel = {
            "contract": marginal["fit_kernel"],
            "replicate_chunk_rule": marginal["replicate_chunk_rule"],
            "replicate_chunk_size": calibration._replicate_chunk_size(sample_count),  # noqa: SLF001
            "scalar_fallback_count": fallback_count,
        }
        resource_usage = task.get("resource_usage")
        expected_task = {
            "schema_version": calibration.SCHEMA_VERSION,
            "contract": calibration.TASK_CONTRACT,
            "cohort": cohort,
            "provider": provider_key,
            "role": cell.role,
            "config_sha256": config_sha256,
            "run_completion_sha256": completion_sha256,
            "seed": calibration._seed(int(config["seed"]), cohort, provider_key),  # noqa: SLF001
            "marginal_replicates": replicates,
            "sentinel_pair_count": pair_count,
            "alphas": marginal["alphas"],
            "replicate_rng": marginal["replicate_rng"],
            "fit_kernel": fit_kernel,
            "worker_topology": calibration._worker_topology(  # noqa: SLF001
                config,
                provider=provider_key,
            ),
            "marginal_reportable_count": int(reportable.sum()),
            "source_task_manifest": source_manifest,
            "single_gene_input": single_gene_input,
            "resource_usage": resource_usage,
            "output": _archive_record(
                records,
                data_name,
                path=calibration.TASK_DATA_NAME,
            ),
        }
        calibration._validate_calibration_resource_usage(  # noqa: SLF001
            expected_task,
            config,
            provider=provider_key,
        )
        core._validate_task_resource_usage(  # noqa: SLF001
            expected_task,
            Path(task_name),
        )
        if task != expected_task:
            msg = (
                "Archived calibration task differs from its evidence: "
                f"{cohort}/{provider_key}"
            )
            raise ValueError(msg)
        relative_task = (
            f"tasks/{cohort}/{provider_key}/{calibration.TASK_MANIFEST_NAME}"
        )
        summary_tasks.append(
            {
                **_archive_record(records, task_name, path=relative_task),
                "cohort": cohort,
                "provider": provider_key,
                "role": cell.role,
                "fit_kernel": fit_kernel,
                "worker_topology": expected_task["worker_topology"],
                "resource_usage": resource_usage,
            },
        )
        log_p_values = calibration._effective_log_p_values(  # noqa: SLF001
            likelihood_ratios,
            reportable,
        )
        nonreportable_fit_count += int(reportable.size - reportable.sum())
        for pair_index in range(pair_count):
            pair_log_p_values = log_p_values[:, pair_index]
            pair_reportable = reportable[:, pair_index]
            reportable_trials = int(pair_reportable.sum())
            for raw_alpha in marginal["alphas"]:
                alpha = float(raw_alpha)
                events = int((pair_log_p_values <= np.log(alpha)).sum())
                row: dict[str, Any] = {
                    "cohort": cohort,
                    "provider": provider_key,
                    "role": cell.role,
                    "screen": calibration.MARGINAL_SCREEN,
                    "sentinel_pair_index": pair_index,
                    "threshold": alpha,
                    "events": events,
                    "trials": replicates,
                    "rate": events / replicates,
                    "reportable_trials": reportable_trials,
                    "nonreportable_trials": replicates - reportable_trials,
                    "gate_endpoint": cell.role == calibration.PRIMARY_ROLE,
                    "exact_binomial_familywise_error": "",
                    "exact_binomial_endpoint_count": "",
                    "bonferroni_endpoint_error": "",
                    "clopper_pearson_upper_bound": "",
                    "acceptance_upper_bound": "",
                    "endpoint_gate_pass": calibration.GATE_NOT_APPLICABLE,
                }
                if cell.role == calibration.PRIMARY_ROLE:
                    row.update(
                        calibration._gate_fields(  # noqa: SLF001
                            successes=events,
                            trials=replicates,
                            alpha=alpha,
                            config=config,
                        ),
                    )
                rows.append(row)

    frame = pd.DataFrame(rows, columns=calibration.SUMMARY_COLUMNS)
    gate_rows = calibration._validated_gate_rows(frame, config)  # noqa: SLF001
    passed = int(
        gate_rows["endpoint_gate_pass"].eq(calibration.ENDPOINT_ACCEPTED).sum(),
    )
    table_bytes = calibration._summary_csv_bytes(frame)  # noqa: SLF001
    archived_table = _verified_member_bytes(
        archive,
        table_name,
        records,
        limit=_JSON_LIMIT_BYTES,
    )
    if archived_table != table_bytes:
        msg = "Archived calibration table differs from exact NPZ recomputation."
        raise ValueError(msg)
    gate = config["affirmative_gate"]
    reporting_candidates = config["reporting_candidates"]
    endpoint_error = float(gate["familywise_error"]) / int(gate["endpoint_count"])
    expected_summary = {
        "schema_version": calibration.SCHEMA_VERSION,
        "contract": calibration.SUMMARY_CONTRACT,
        "config_sha256": config_sha256,
        "cell_count": len(protocol),
        "primary_gate_cell_count": sum(
            cell.role == calibration.PRIMARY_ROLE for cell in protocol
        ),
        "descriptive_cell_count": sum(
            cell.role == calibration.DESCRIPTIVE_ROLE for cell in protocol
        ),
        "marginal_endpoint_count": len(frame),
        "primary_gate_endpoint_count": len(gate_rows),
        "primary_gate_passed_endpoint_count": passed,
        "overall_gate_pass": passed == len(gate_rows),
        "gate_provider": gate["provider"],
        "gate_endpoint_unit": gate["endpoint_unit"],
        "gate_method": gate["method"],
        "exact_binomial_familywise_error": gate["familywise_error"],
        "exact_binomial_endpoint_count": gate["endpoint_count"],
        "bonferroni_endpoint_error": endpoint_error,
        "clopper_pearson_confidence_level": 1.0 - endpoint_error,
        "acceptance_upper_bounds": gate["acceptance_upper_bounds"],
        "effective_p_policy": "chi-square-one-df-for-full-affine-rank-otherwise-p-one",
        "nonreportable_fit_count": nonreportable_fit_count,
        "primary_adjustment": reporting_candidates["primary_adjustment"],
        "primary_q_candidate": reporting_candidates["primary_q_threshold"],
        "sensitivity_adjustment": reporting_candidates["sensitivity_adjustment"],
        "sensitivity_q_candidate": reporting_candidates["sensitivity_q_threshold"],
        "interpretation": reporting_candidates["interpretation"],
        "reporting_rule_selected": False,
        "resource_contract": dict(config["resources"]),
        "runtime_resource_observation": expected_run["runtime_resource_observation"],
        "thread_environment": dict(calibration.THREAD_ENV),
        "resource_usage_interpretation": {
            "self_and_terminated_child_cpu_seconds_reported_separately": True,
            "terminated_child_peak_rss": (
                "maximum-over-terminated-children-not-additive"
            ),
        },
        "result_blindness": calibration._result_blindness_receipt(),  # noqa: SLF001
        "run_completion_sha256": completion_sha256,
        "provider_manifest_sha256": provider_sha256,
        "run_manifest": _archive_record(records, run_name, path="run_manifest.json"),
        "task_manifests": summary_tasks,
        "table": _archive_record(
            records,
            table_name,
            path=calibration.SUMMARY_TABLE_NAME,
        ),
    }
    summary = _verified_json_member(archive, summary_name, records)
    if summary != expected_summary:
        msg = "Archived calibration summary differs from exact NPZ recomputation."
        raise ValueError(msg)
    if expected_summary["overall_gate_pass"] is not True:
        msg = (
            "Association-level archive verification is withheld: "
            "recomputed gate failed."
        )
        raise RuntimeError(msg)
    rule = _verified_json_member(archive, REPORTING_RULE_MEMBER, records)
    expected_rule = {
        "schema_version": rule_module.SCHEMA_VERSION,
        "contract": rule_module.RULE_CONTRACT,
        "analysis_config_sha256": analysis_config_sha256,
        "calibration_config_sha256": config_sha256,
        "calibration_summary_sha256": str(records[summary_name]["sha256"]),
        "postprocess_manifest_sha256": str(
            records["results/postprocess/postprocess_manifest.json"]["sha256"],
        ),
        "scope": "one-identical-rule-across-all-32-tcga-pan-cancer-atlas-cohorts",
        "test": "chi-square-one-df-profile-lrt",
        "effective_p_policy": (
            "chi-square-one-df-for-full-affine-rank-otherwise-p-one"
        ),
        "multiplicity": "provider-specific-complete-within-cohort-family",
        "primary_adjustment": "benjamini-yekutieli",
        "sensitivity_adjustment": "benjamini-hochberg",
        "primary_provider": "mutsig",
        "continuity_provider": "cbase",
        "supplementary_providers": ["dig"],
        "primary_q_threshold": 0.01,
        "sensitivity_q_threshold": 0.01,
        "threshold_comparison": "inclusive-less-than-or-equal",
        "direction": "primary-provider-rho-sign-after-nondirectional-rejection",
        "direction_unavailable": (
            "retain-nondirectional-rejection-exclude-from-me-co-lists"
        ),
        "provider_overlap": "descriptive-only-not-an-inferential-vote",
        "me_presentation": "primary-MutSig-with-CBaSE-continuity-comparison",
        "co_presentation": "primary-MutSig-with-CBaSE-and-DIG-sensitivity",
        "thresholds_selected_from_observed_pairs": False,
        "calibration_gate": {
            "provider": expected_summary["gate_provider"],
            "endpoint_unit": expected_summary["gate_endpoint_unit"],
            "method": expected_summary["gate_method"],
            "endpoint_count": expected_summary["exact_binomial_endpoint_count"],
            "familywise_error": expected_summary[
                "exact_binomial_familywise_error"
            ],
            "acceptance_upper_bounds": expected_summary[
                "acceptance_upper_bounds"
            ],
            "overall_gate_pass": True,
        },
        "inference_status": rule_module.REPORTABLE_STATUS,
        "withheld_reason": None,
        "claim_scope": "finite-scenario-calibrated-nominal-inference",
        "calibration_interpretation": (
            "finite-scenario-stress-not-formal-uniform-FDR-proof"
        ),
    }
    if rule != expected_rule:
        msg = "Archived reporting rule differs from the frozen reportable contract."
        raise ValueError(msg)
    return expected_summary


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
    ordered_columns = tuple(
        column.strip()
        for column in next(csv.reader([header.decode("utf-8-sig")]), [])
    )
    columns = {column.casefold() for column in ordered_columns}
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
    member = PurePosixPath(name)
    if member.parts[:2] == ("results", "report"):
        expected = reporting.report_csv_columns().get(member.name)
        if expected is None or ordered_columns != expected:
            msg = f"Archived report CSV violates its exact public schema: {name}"
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
    if (
        not isinstance(record, dict)
        or set(record) != {"path", "bytes", "sha256"}
        or record.get("path") != expected_path
    ):
        msg = f"Release semantic receipt path is broken at {label}."
        raise ValueError(msg)
    _require_record(record, records, member_name, label=label)


def _valid_source_records(
    value: object,
    *,
    expected_paths: tuple[str, ...],
) -> bool:
    if not isinstance(value, list) or len(value) != len(expected_paths):
        return False
    return all(
        isinstance(record, dict)
        and set(record) == {"path", "bytes", "sha256"}
        and record.get("path") == path
        and isinstance(record.get("bytes"), int)
        and not isinstance(record.get("bytes"), bool)
        and int(record["bytes"]) >= 0
        and re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256", "")))
        is not None
        for record, path in zip(value, expected_paths, strict=True)
    )


def _valid_detached_record(
    value: object,
    *,
    expected_path: str,
    allow_zero_bytes: bool = False,
) -> bool:
    """Validate a receipt for source bytes intentionally absent from the archive."""
    minimum_bytes = 0 if allow_zero_bytes else 1
    return (
        isinstance(value, dict)
        and set(value) == {"path", "bytes", "sha256"}
        and value.get("path") == expected_path
        and isinstance(value.get("bytes"), int)
        and not isinstance(value.get("bytes"), bool)
        and int(value["bytes"]) >= minimum_bytes
        and re.fullmatch(r"[0-9a-f]{64}", str(value.get("sha256", "")))
        is not None
    )


def _valid_hash(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _validate_raw_manifest_schemas(
    *,
    input_manifest: Mapping[str, Any],
    provider_manifest: Mapping[str, Any],
    run_manifest: Mapping[str, Any],
    completion_manifest: Mapping[str, Any],
    analysis_config_sha256: str,
) -> dict[str, Mapping[str, Any]]:
    """Validate exact schemas for the receipt-only raw execution chain."""
    input_records_value = input_manifest.get("cohort_records")
    input_record_items = (
        input_records_value if isinstance(input_records_value, list) else []
    )
    input_records = {
        str(record.get("cohort")): record
        for record in input_record_items
        if isinstance(record, dict)
    }
    if (
        set(input_manifest)
        != {
            "schema_version",
            "contract",
            "config",
            "config_sha256",
            "datahub_commit",
            "population_manifest",
            "cohorts",
            "cohort_count",
            "participant_count",
            "cohort_records",
        }
        or input_manifest.get("schema_version") != preparation.SCHEMA_VERSION
        or input_manifest.get("contract") != preparation.INPUT_CONTRACT
        or input_manifest.get("config_sha256") != analysis_config_sha256
        or input_manifest.get("datahub_commit") != preparation.TCGA_DATAHUB_COMMIT
        or input_manifest.get("cohorts") != list(TCGA_COHORTS)
        or input_manifest.get("cohort_count") != len(TCGA_COHORTS)
        or input_manifest.get("participant_count") != reporting.EXPECTED_TUMOR_COUNT
        or not _valid_detached_record(
            input_manifest.get("population_manifest"),
            expected_path="population-source/population_manifest.json",
        )
        or not isinstance(input_records_value, list)
        or len(input_records_value) != len(TCGA_COHORTS)
        or list(input_records) != list(TCGA_COHORTS)
    ):
        msg = "Archived focused input manifest violates its exact schema."
        raise ValueError(msg)
    expected_record_keys = {
        "canonical_maf",
        "cohort",
        "duplicate_resolution_policy",
        "raw_maf",
        "row_accounting",
        "sample_axis_sha256",
        "sample_count",
    }
    expected_row_keys = {
        "raw_rows",
        "selected_rows",
        "canonical_rows",
        "removed_duplicate_rows",
        "multiallelic_groups_preserved",
        "unresolved_semantic_conflicts",
    }
    expected_duplicate_policy = json.loads(
        _canonical_json(
            preparation.asdict(
                preparation.variant_data.TCGA_DUPLICATE_RESOLUTION_POLICY,
            ),
        ),
    )
    sample_total = 0
    for cohort in TCGA_COHORTS:
        record = input_records[cohort]
        rows = record.get("row_accounting")
        raw_maf = record.get("raw_maf")
        sample_count = record.get("sample_count")
        if (
            set(record) != expected_record_keys
            or record.get("cohort") != cohort
            or not _valid_detached_record(
                record.get("canonical_maf"),
                expected_path=f"mafs/{cohort}.maf",
            )
            or not isinstance(raw_maf, dict)
            or set(raw_maf) != {"bytes", "sha256"}
            or not isinstance(raw_maf.get("bytes"), int)
            or isinstance(raw_maf.get("bytes"), bool)
            or int(raw_maf["bytes"]) <= 0
            or not _valid_hash(raw_maf.get("sha256"))
            or not isinstance(rows, dict)
            or set(rows) != expected_row_keys
            or not all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for value in rows.values()
            )
            or rows["raw_rows"] < rows["selected_rows"]
            or rows["canonical_rows"] > rows["selected_rows"]
            or rows["removed_duplicate_rows"]
            != rows["selected_rows"] - rows["canonical_rows"]
            or rows["unresolved_semantic_conflicts"] != 0
            or not _valid_hash(record.get("sample_axis_sha256"))
            or not isinstance(sample_count, int)
            or isinstance(sample_count, bool)
            or sample_count <= 0
            or record.get("duplicate_resolution_policy")
            != expected_duplicate_policy
        ):
            msg = f"Archived focused input cohort record is invalid: {cohort}"
            raise ValueError(msg)
        sample_total += sample_count
    if sample_total != reporting.EXPECTED_TUMOR_COUNT:
        msg = "Archived focused input sample counts do not reconcile."
        raise ValueError(msg)

    provider_records_value = provider_manifest.get("records")
    provider_record_items = (
        provider_records_value if isinstance(provider_records_value, list) else []
    )
    provider_records = {
        str(record.get("cohort")): record
        for record in provider_record_items
        if isinstance(record, dict)
    }
    if (
        set(provider_manifest)
        != {
            "schema_version",
            "contract",
            "config_sha256",
            "input_manifest",
            "cohorts",
            "cohort_count",
            "records",
        }
        or provider_manifest.get("schema_version") != preparation.SCHEMA_VERSION
        or provider_manifest.get("contract") != preparation.PROVIDER_CONTRACT
        or provider_manifest.get("config_sha256") != analysis_config_sha256
        or provider_manifest.get("cohorts") != list(TCGA_COHORTS)
        or provider_manifest.get("cohort_count") != len(TCGA_COHORTS)
        or not isinstance(provider_records_value, list)
        or len(provider_records_value) != len(TCGA_COHORTS)
        or list(provider_records) != list(TCGA_COHORTS)
    ):
        msg = "Archived focused provider manifest violates its exact schema."
        raise ValueError(msg)
    for cohort in TCGA_COHORTS:
        record = provider_records[cohort]
        files = record.get("files")
        mutsig_files = record.get("mutsig_files")
        if (
            set(record) != {"cohort", "files", "mutsig_files"}
            or record.get("cohort") != cohort
            or not isinstance(files, dict)
            or set(files) != set(preparation.REQUIRED_PROVIDER_FILES)
            or not isinstance(mutsig_files, dict)
            or set(mutsig_files) != set(preparation.REQUIRED_MUTSIG_FILES)
            or any(
                not _valid_detached_record(
                    files[name],
                    expected_path=f"cohorts/{cohort}/{name}",
                )
                for name in preparation.REQUIRED_PROVIDER_FILES
            )
            or any(
                not _valid_detached_record(
                    mutsig_files[name],
                    expected_path=f"mutsig/{cohort}/{name}",
                )
                for name in preparation.REQUIRED_MUTSIG_FILES
            )
        ):
            msg = f"Archived focused provider record is invalid: {cohort}"
            raise ValueError(msg)
        if any(
            files["sample_axis.txt"][key]
            != mutsig_files["persample_patients.txt"][key]
            for key in ("bytes", "sha256")
        ):
            msg = f"Archived MutSig and count sample axes differ: {cohort}"
            raise ValueError(msg)

    analysis_config = preparation._load_config()  # noqa: SLF001
    if (
        set(run_manifest)
        != {
            "schema_version",
            "contract",
            "config",
            "config_sha256",
            "provider_manifest",
            "cohorts",
            "providers",
            "top_k",
            "resources",
        }
        or run_manifest.get("schema_version") != provenance.runner.SCHEMA_VERSION
        or run_manifest.get("contract") != provenance.runner.RUN_CONTRACT
        or run_manifest.get("config_sha256") != analysis_config_sha256
        or run_manifest.get("cohorts") != list(TCGA_COHORTS)
        or run_manifest.get("providers") != list(core.BMRS)
        or run_manifest.get("top_k") != 500
        or run_manifest.get("resources") != analysis_config["execution"]
    ):
        msg = "Archived focused run manifest violates its exact schema."
        raise ValueError(msg)
    if (
        set(completion_manifest)
        != {
            "schema_version",
            "contract",
            "config_sha256",
            "run_manifest",
            "cohorts",
            "task_count",
            "tasks",
        }
        or completion_manifest.get("schema_version")
        != provenance.runner.SCHEMA_VERSION
        or completion_manifest.get("contract")
        != provenance.runner.COMPLETION_CONTRACT
        or completion_manifest.get("config_sha256") != analysis_config_sha256
        or completion_manifest.get("cohorts") != list(TCGA_COHORTS)
        or completion_manifest.get("task_count") != len(TCGA_COHORTS) * len(core.BMRS)
    ):
        msg = "Archived focused completion manifest violates its exact schema."
        raise ValueError(msg)
    return input_records


def _validate_archived_raw_task(  # noqa: PLR0913
    task: Mapping[str, Any],
    *,
    cohort: str,
    provider: str,
    analysis_config_sha256: str,
    contract_sha256: str,
    pair_count: int,
) -> None:
    """Validate one exact raw-task manifest before trusting aggregate fields."""
    outputs = task.get("outputs")
    usage = task.get("resource_usage")
    peak = usage.get("peak_rss") if isinstance(usage, dict) else None
    if (
        set(task)
        != {
            "schema_version",
            "contract",
            "cohort",
            "provider",
            "top_k",
            "contract_sha256",
            "config_sha256",
            "single_gene_rows",
            "pairwise_rows",
            "resource_usage",
            "outputs",
        }
        or task.get("schema_version") != provenance.runner.SCHEMA_VERSION
        or task.get("contract") != provenance.runner.TASK_CONTRACT
        or task.get("cohort") != cohort
        or task.get("provider") != provider
        or task.get("top_k") != 500
        or task.get("contract_sha256") != contract_sha256
        or task.get("config_sha256") != analysis_config_sha256
        or task.get("single_gene_rows") != 500
        or task.get("pairwise_rows") != pair_count
        or not isinstance(outputs, dict)
        or set(outputs)
        != {"pairwise_interaction_results.csv", "single_gene_results.csv"}
        or any(
            not _valid_detached_record(outputs[name], expected_path=name)
            for name in outputs
        )
        or not isinstance(usage, dict)
        or set(usage)
        != {
            "elapsed_seconds",
            "user_cpu_seconds",
            "system_cpu_seconds",
            "peak_rss",
        }
        or any(
            not isinstance(usage.get(name), (int, float))
            or isinstance(usage.get(name), bool)
            or not np.isfinite(float(usage[name]))
            or float(usage[name]) < 0
            for name in ("user_cpu_seconds", "system_cpu_seconds")
        )
        or not isinstance(peak, dict)
        or set(peak)
        != {"bytes", "native_value", "native_unit", "platform", "source"}
    ):
        msg = f"Archived raw task manifest contract is invalid: {cohort}/{provider}"
        raise ValueError(msg)
    core._validate_task_resource_usage(dict(task), Path(f"{cohort}/{provider}"))  # noqa: SLF001


def _validate_attestation_schema(
    attestation: Mapping[str, Any],
    source_record: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    """Validate the exact public source/runtime attestation contract."""
    expected_privacy = {
        "raw_tumor_level_inputs_included": False,
        "sample_identifiers_included": False,
        "restricted_mutsig_source_included": False,
    }
    source = attestation.get("source")
    runtime = attestation.get("runtime")
    raw_chain = attestation.get("raw_chain")
    expected_fit_paths = tuple(
        path.as_posix()
        for path in sorted(
            provenance.FIT_SOURCE_FILES,
            key=lambda item: item.as_posix(),
        )
    )
    expected_release_paths = tuple(
        path.as_posix()
        for path in sorted(
            provenance.RELEASE_PIPELINE_FILES,
            key=lambda item: item.as_posix(),
        )
    )
    expected_excluded = [
        {
            "path": "src/dialect/_version.py",
            "reason": (
                "setuptools-scm generated module absent from both Git commits; "
                "not independently attributable to the completed task runtime"
            ),
        },
    ]
    valid_source = (
        isinstance(source, dict)
        and set(source)
        == {
            "fit_source_commit",
            "fit_source_tree",
            "release_source_commit",
            "release_source_tree",
            "fit_is_ancestor_of_release",
            "fit_source_files",
            "excluded_generated_fit_sources",
            "raw_fit_sources_unchanged_at_release",
            "release_pipeline_files",
            "repository",
        }
        and source.get("fit_source_commit") == manifest.get("fit_source_commit")
        and source.get("release_source_commit") == manifest.get("release_source_commit")
        and re.fullmatch(r"[0-9a-f]{40}", str(source.get("fit_source_tree", "")))
        is not None
        and re.fullmatch(r"[0-9a-f]{40}", str(source.get("release_source_tree", "")))
        is not None
        and source.get("fit_is_ancestor_of_release") is True
        and source.get("raw_fit_sources_unchanged_at_release") is True
        and source.get("repository") == "raphael-group/dialect"
        and source.get("excluded_generated_fit_sources") == expected_excluded
        and _valid_source_records(
            source.get("fit_source_files"),
            expected_paths=expected_fit_paths,
        )
        and _valid_source_records(
            source.get("release_pipeline_files"),
            expected_paths=expected_release_paths,
        )
    )
    python = runtime.get("python") if isinstance(runtime, dict) else None
    platform_record = runtime.get("platform") if isinstance(runtime, dict) else None
    packages = runtime.get("packages") if isinstance(runtime, dict) else None
    valid_runtime = (
        isinstance(runtime, dict)
        and set(runtime)
        == {"scope", "python", "platform", "packages", "thread_environment"}
        and runtime.get("scope")
        == "post-run-runtime-readback-not-process-memory-attestation"
        and isinstance(python, dict)
        and set(python)
        == {
            "basename",
            "bytes",
            "sha256",
            "version",
            "implementation",
            "cache_tag",
        }
        and isinstance(python.get("basename"), str)
        and bool(python["basename"])
        and isinstance(python.get("bytes"), int)
        and not isinstance(python.get("bytes"), bool)
        and int(python["bytes"]) > 0
        and re.fullmatch(r"[0-9a-f]{64}", str(python.get("sha256", "")))
        is not None
        and all(
            isinstance(python.get(key), str) and bool(python[key])
            for key in ("version", "implementation", "cache_tag")
        )
        and isinstance(platform_record, dict)
        and set(platform_record) == {"system", "release", "machine", "byteorder"}
        and all(
            isinstance(platform_record.get(key), str) and bool(platform_record[key])
            for key in ("system", "release", "machine", "byteorder")
        )
        and isinstance(packages, dict)
        and set(packages) == set(provenance.RUNTIME_DISTRIBUTIONS)
        and all(
            isinstance(version, str) and bool(version)
            for version in packages.values()
        )
        and runtime.get("thread_environment")
        == dict(sorted(provenance.runner.THREAD_ENV.items()))
    )
    valid_raw_chain = isinstance(raw_chain, dict) and set(raw_chain) == {
        "input_manifest",
        "provider_manifest",
        "run_manifest",
        "completion_manifest",
        "cohort_contracts",
        "task_manifests",
    }
    expected_source_record = {
        "schema_version": SCHEMA_VERSION,
        "contract": SOURCE_RECORD_CONTRACT,
        "repository": "raphael-group/dialect",
        "fit_source_commit": manifest.get("fit_source_commit"),
        "release_source_commit": manifest.get("release_source_commit"),
        "fit_is_ancestor_of_release": True,
        "raw_fit_sources_unchanged_at_release": True,
        "restricted_mutsig_source_included": False,
        "raw_tumor_level_inputs_included": False,
        "sample_identifiers_included": False,
    }
    if (
        set(attestation)
        != {
            "schema_version",
            "contract",
            "scope",
            "source",
            "runtime",
            "raw_chain",
            "privacy",
        }
        or attestation.get("schema_version") != provenance.SCHEMA_VERSION
        or attestation.get("contract") != provenance.FIT_ATTESTATION_CONTRACT
        or attestation.get("scope")
        != (
            "post-run-source-runtime-and-receipt-reconstruction; "
            "not-loaded-process-memory-attestation"
        )
        or attestation.get("privacy") != expected_privacy
        or not valid_source
        or not valid_runtime
        or not valid_raw_chain
        or source_record != expected_source_record
    ):
        msg = "Release fit/release source-runtime attestation schema is invalid."
        raise ValueError(msg)


def _validate_archived_inference_frame(
    frame: pd.DataFrame,
    *,
    cohort: str,
    features: Sequence[str],
) -> None:
    """Recompute inference columns and the complete ordered pair axis."""
    postprocess.validate_inference_frame(frame, cohort=cohort)
    expected_count = int(core._pair_contract(features)["row_count"])  # noqa: SLF001
    if len(frame) != expected_count:
        msg = f"Archived inference has an incomplete pair family: {cohort}"
        raise ValueError(msg)
    observed = zip(
        frame["gene_a"].astype(str),
        frame["gene_b"].astype(str),
        strict=True,
    )
    if any(
        actual != expected
        for actual, expected in zip(
            observed,
            core.iter_tested_pairs(features),
            strict=True,
        )
    ):
        msg = f"Archived inference pair axis differs from its contract: {cohort}"
        raise ValueError(msg)


def _archived_report_frames(
    archive: tarfile.TarFile,
    records: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, pd.DataFrame], dict[str, bytes]]:
    """Load each receipt-verified public report CSV under its exact schema."""
    frames: dict[str, pd.DataFrame] = {}
    payloads: dict[str, bytes] = {}
    for filename, columns in reporting.report_csv_columns().items():
        member_name = f"results/report/{filename}"
        payload = _verified_member_bytes(
            archive,
            member_name,
            records,
            limit=_INFERENCE_LIMIT_BYTES,
        )
        frame = pd.read_csv(BytesIO(payload), float_precision="round_trip")
        if tuple(frame.columns) != columns:
            msg = f"Archived report CSV violates its exact schema: {filename}"
            raise ValueError(msg)
        frames[filename] = frame
        payloads[filename] = payload
    return frames, payloads


def _expected_summary_decisions(
    *,
    cohort: str,
    frame: pd.DataFrame,
    features: Sequence[str],
    sample_count: int,
) -> dict[str, int | float | str]:
    """Reconstruct every Table S5 field available from public release inputs."""
    pair_policy = core._pair_contract(features)  # noqa: SLF001
    expected: dict[str, int | float | str] = {
        "cohort": cohort,
        "primary_adjustment": "BY",
        "primary_q_threshold": 0.01,
        "sensitivity_adjustment": "BH",
        "sensitivity_q_threshold": 0.01,
        "tumors": sample_count,
        "selected_features": len(features),
        "tested_pairs": len(frame),
        "same_base_pairs_excluded": pair_policy["same_base_pairs_excluded"],
        "unfiltered_pair_count": (
            len(frame) + pair_policy["same_base_pairs_excluded"]
        ),
    }
    for provider in core.BMRS:
        directions = frame[f"{provider}_direction"].astype("string")
        for label, adjustment in (
            ("primary", "benjamini-yekutieli"),
            ("sensitivity", "benjamini-hochberg"),
        ):
            crossing = reporting._threshold_crossing(  # noqa: SLF001
                frame,
                provider,
                adjustment,
                0.01,
            )
            prefix = reporting._decision_prefix(provider, label)  # noqa: SLF001
            expected[f"{prefix}_total"] = int(crossing.sum())
            expected[f"{prefix}_me"] = int(
                (crossing & directions.eq("ME").to_numpy()).sum(),
            )
            expected[f"{prefix}_co"] = int(
                (crossing & directions.eq("CO").to_numpy()).sum(),
            )
            expected[f"{prefix}_direction_unavailable"] = int(
                (crossing & ~directions.isin(["ME", "CO"]).to_numpy()).sum(),
            )
    return expected


def _validate_archived_burden_summary(summary: pd.DataFrame) -> None:
    """Validate the irreversibly aggregated burden fields without sample rows."""
    burden_columns = [
        "burden_median",
        "burden_q25",
        "burden_q75",
        "burden_p90",
        "burden_p95",
        "burden_max",
        "high_burden_fraction",
    ]
    values = summary.loc[:, burden_columns].to_numpy(dtype=np.float64)
    fractions = summary["high_burden_fraction"].to_numpy(dtype=np.float64)
    ordered = summary.loc[
        :,
        [
            "burden_q25",
            "burden_median",
            "burden_q75",
            "burden_p90",
            "burden_p95",
            "burden_max",
        ],
    ].to_numpy(dtype=np.float64)
    if (
        not np.isfinite(values).all()
        or (values[:, :-1] < 0).any()
        or (fractions < 0).any()
        or (fractions > 1).any()
        or (np.diff(ordered, axis=1) < 0).any()
    ):
        msg = "Archived Table S5 has invalid aggregate burden summaries."
        raise ValueError(msg)


def _validate_archived_figure6_bins(
    frame: pd.DataFrame,
    *,
    sample_count: int,
) -> None:
    """Validate aggregate-only Figure 6 burden bins and their shared tumor axis."""
    if (
        frame.empty
        or not frame["cohort"].eq(reporting.FOCAL_BURDEN_COHORT).all()
        or set(frame["provider"].astype(str)) != set(core.BMRS)
        or frame.duplicated(
            [
                "provider",
                "observed_log1p_bin_lower",
                "observed_log1p_bin_upper",
                "expected_log1p_bin_lower",
                "expected_log1p_bin_upper",
            ],
        ).any()
    ):
        msg = "Archived Figure 6 burden bins have an invalid aggregate axis."
        raise ValueError(msg)
    numeric_columns = list(reporting.BURDEN_BIN_COLUMNS[2:])
    numeric = frame.loc[:, numeric_columns].to_numpy(dtype=np.float64)
    counts = frame["tumor_count"].to_numpy(dtype=np.float64)
    if (
        not np.isfinite(numeric).all()
        or (counts <= 0).any()
        or not np.equal(counts, np.floor(counts)).all()
    ):
        msg = "Archived Figure 6 burden bins contain invalid values."
        raise ValueError(msg)
    edges = np.linspace(
        0.0,
        reporting.BURDEN_LOG1P_MAX,
        reporting.BURDEN_BIN_COUNT + 1,
    )
    for lower_column, upper_column in (
        ("observed_log1p_bin_lower", "observed_log1p_bin_upper"),
        ("expected_log1p_bin_lower", "expected_log1p_bin_upper"),
    ):
        lower = frame[lower_column].to_numpy(dtype=np.float64)
        upper = frame[upper_column].to_numpy(dtype=np.float64)
        indices = np.searchsorted(edges, lower, side="left")
        valid_indices = indices < len(edges) - 1
        if (
            not valid_indices.all()
            or not np.allclose(lower, edges[indices], rtol=0.0, atol=1e-12)
            or not np.allclose(upper, edges[indices + 1], rtol=0.0, atol=1e-12)
        ):
            msg = "Archived Figure 6 burden bins differ from the frozen grid."
            raise ValueError(msg)
    totals = frame.groupby("provider", sort=False)["tumor_count"].sum()
    if any(int(totals.get(provider, -1)) != sample_count for provider in core.BMRS):
        msg = "Archived Figure 6 burden bins do not cover the focal tumor set."
        raise ValueError(msg)
    observed_axes = []
    for provider in core.BMRS:
        aggregate = (
            frame.loc[frame["provider"].eq(provider)]
            .groupby(
                ["observed_log1p_bin_lower", "observed_log1p_bin_upper"],
                as_index=False,
                sort=True,
            )["tumor_count"]
            .sum()
        )
        observed_axes.append(aggregate)
    if any(not observed_axes[0].equals(other) for other in observed_axes[1:]):
        msg = "Archived Figure 6 providers do not share one observed tumor axis."
        raise ValueError(msg)


def _validate_archived_report_derivations(  # noqa: PLR0913
    archive: tarfile.TarFile,
    records: Mapping[str, Mapping[str, Any]],
    *,
    expected_summary_rows: Mapping[str, Mapping[str, int | float | str]],
    expected_overlap: pd.DataFrame,
    expected_top: pd.DataFrame,
    sample_counts: Mapping[str, int],
    raw_tasks: Mapping[tuple[str, str], Mapping[str, Any]],
    cohort_diagnostics: Mapping[str, Mapping[str, Any]],
    calibration_table_name: str,
    high_burden_threshold: float,
) -> None:
    """Reconstruct all public report claims supported by archived inputs."""
    frames, payloads = _archived_report_frames(archive, records)
    summary = frames["table_s5.csv"]
    if summary["cohort"].astype(str).tolist() != list(TCGA_COHORTS):
        msg = "Archived Table S5 does not have the exact ordered cohort axis."
        raise ValueError(msg)
    for row, cohort in zip(
        summary.to_dict(orient="records"),
        TCGA_COHORTS,
        strict=True,
    ):
        expected = expected_summary_rows[cohort]
        if any(row.get(column) != value for column, value in expected.items()):
            msg = f"Archived Table S5 differs from source recomputation: {cohort}"
            raise ValueError(msg)
    _validate_archived_burden_summary(summary)
    reporting._validate_burden_histogram_summary(  # noqa: SLF001
        frames["cohort_burden_histogram.csv"],
        summary,
        high_burden_threshold,
    )
    if int(summary["tumors"].sum()) != reporting.EXPECTED_TUMOR_COUNT:
        msg = "Archived Table S5 does not cover the frozen tumor population."
        raise ValueError(msg)

    if (
        payloads["provider_overlap.csv"] != reporting._csv_bytes(expected_overlap)  # noqa: SLF001
        or payloads["top_primary_pairs.csv"] != reporting._csv_bytes(expected_top)  # noqa: SLF001
    ):
        msg = "Archived interaction summaries differ from inference recomputation."
        raise ValueError(msg)

    runtime_rows = []
    for cohort in TCGA_COHORTS:
        for provider in core.BMRS:
            task = raw_tasks[(cohort, provider)]
            usage = task.get("resource_usage", {})
            peak = usage.get("peak_rss", {}) if isinstance(usage, dict) else {}
            runtime_rows.append(
                {
                    "cohort": cohort,
                    "provider": provider,
                    "pairwise_rows": task.get("pairwise_rows"),
                    "elapsed_seconds": usage.get("elapsed_seconds"),
                    "user_cpu_seconds": usage.get("user_cpu_seconds"),
                    "system_cpu_seconds": usage.get("system_cpu_seconds"),
                    "peak_rss_bytes": peak.get("bytes"),
                },
            )
    expected_runtime = pd.DataFrame(runtime_rows, columns=reporting.RUNTIME_COLUMNS)
    if payloads["runtime_summary.csv"] != reporting._csv_bytes(expected_runtime):  # noqa: SLF001
        msg = "Archived runtime summary differs from raw task manifests."
        raise ValueError(msg)

    fit = frames["fit_diagnostics_summary.csv"]
    if fit["scope"].astype(str).tolist() != ["all", *core.BMRS]:
        msg = "Archived fit diagnostics have an invalid scope axis."
        raise ValueError(msg)
    expected_by_provider: dict[str, dict[str, int]] = {}
    for provider in core.BMRS:
        expected_by_provider[provider] = {
            "pairwise_rows": sum(
                int(raw_tasks[(cohort, provider)]["pairwise_rows"])
                for cohort in TCGA_COHORTS
            ),
            "full_affine_rank_rows": sum(
                int(cohort_diagnostics[cohort][provider]["full_affine_rank_count"])
                for cohort in TCGA_COHORTS
            ),
            "rank_deficient_rows": sum(
                int(cohort_diagnostics[cohort][provider]["rank_deficient_count"])
                for cohort in TCGA_COHORTS
            ),
            "rank_not_certified_underflow_rows": sum(
                int(
                    cohort_diagnostics[cohort][provider][
                        "rank_not_certified_underflow_count"
                    ],
                )
                for cohort in TCGA_COHORTS
            ),
        }
    for row in fit.to_dict(orient="records"):
        scope = str(row["scope"])
        providers = core.BMRS if scope == "all" else (scope,)
        for column in (
            "pairwise_rows",
            "full_affine_rank_rows",
            "rank_deficient_rows",
            "rank_not_certified_underflow_rows",
        ):
            expected = sum(
                expected_by_provider[provider][column] for provider in providers
            )
            if row[column] != expected:
                msg = f"Archived fit diagnostics differ from receipts: {scope}/{column}"
                raise ValueError(msg)
        if (
            row["converged_rows"] + row["nonconverged_rows"] != row["pairwise_rows"]
            or row["full_affine_rank_rows"]
            + row["rank_deficient_rows"]
            + row["rank_not_certified_underflow_rows"]
            != row["pairwise_rows"]
        ):
            msg = f"Archived fit diagnostic counts do not reconcile: {scope}"
            raise ValueError(msg)
        numeric = np.asarray(
            [
                row["iterations_min"],
                row["iterations_median"],
                row["iterations_p95"],
                row["iterations_max"],
                row["minimum_last_ll_gain"],
                row["maximum_last_ll_gain"],
                row["maximum_fixed_point_residual"],
                row["maximum_kkt_residual"],
            ],
            dtype=np.float64,
        )
        if (
            not np.isfinite(numeric).all()
            or (numeric[:4] < 0).any()
            or not np.all(np.diff(numeric[:4]) >= 0)
            or numeric[4] < -1e-12
            or numeric[5] < numeric[4]
            or (numeric[6:] < 0).any()
        ):
            msg = f"Archived fit diagnostic values are invalid: {scope}"
            raise ValueError(msg)

    figure_bins = frames["figure6_burden_bins.csv"]
    _validate_archived_figure6_bins(
        figure_bins,
        sample_count=sample_counts[reporting.FOCAL_BURDEN_COHORT],
    )
    expected_tex = reporting._table_s5_tex(  # noqa: SLF001
        summary,
        primary_adjustment="benjamini-yekutieli",
        primary_q=0.01,
        sensitivity_adjustment="benjamini-hochberg",
        sensitivity_q=0.01,
    ).encode()
    table_tex = _verified_member_bytes(
        archive,
        "results/report/table_s5.tex",
        records,
        limit=_JSON_LIMIT_BYTES,
    )
    if table_tex != expected_tex:
        msg = "Archived Table S5 LaTeX differs from its validated CSV."
        raise ValueError(msg)

    calibration_table = pd.read_csv(
        BytesIO(
            _verified_member_bytes(
                archive,
                calibration_table_name,
                records,
                limit=_INFERENCE_LIMIT_BYTES,
            ),
        ),
        float_precision="round_trip",
    )
    with reporting.tempfile.TemporaryDirectory(prefix="dialect-archive-figure-") as tmp:
        expected_figure = Path(tmp) / "figure6.pdf"
        reporting._plot_figure6(  # noqa: SLF001
            burden_bins=figure_bins,
            summary=summary,
            overlap=expected_overlap,
            calibration_table=calibration_table,
            primary_adjustment="benjamini-yekutieli",
            primary_q=0.01,
            output=expected_figure,
        )
        archived_figure = _verified_member_bytes(
            archive,
            "results/report/figure6.pdf",
            records,
            limit=_INFERENCE_LIMIT_BYTES,
        )
        if archived_figure != expected_figure.read_bytes():
            msg = "Archived Figure 6 differs from its validated source tables."
            raise ValueError(msg)


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
    _validate_attestation_schema(attestation, source, manifest)
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
    input_records = _validate_raw_manifest_schemas(
        input_manifest=input_manifest,
        provider_manifest=provider,
        run_manifest=run,
        completion_manifest=completion,
        analysis_config_sha256=str(analysis_config_sha256),
    )
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
    _require_record_path(
        provider.get("input_manifest"),
        records,
        input_name,
        expected_path="input_manifest.json",
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
    _require_record_path(
        run.get("provider_manifest"),
        records,
        provider_name,
        expected_path="provider_manifest.json",
        label="run to provider",
    )
    _require_record_path(
        completion.get("run_manifest"),
        records,
        run_name,
        expected_path="run_manifest.json",
        label="completion to run",
    )
    expected_coordinates = {
        (cohort, provider_key)
        for cohort in TCGA_COHORTS
        for provider_key in ("cbase", "dig", "mutsig")
    }
    tasks = completion.get("tasks")
    if not isinstance(tasks, list) or any(
        not isinstance(task, dict)
        or set(task) != {"cohort", "provider", "manifest"}
        for task in tasks
    ):
        msg = "Release completion task records violate their exact schema."
        raise ValueError(msg)
    coordinates = {
        (str(task.get("cohort")), str(task.get("provider")))
        for task in tasks
    }
    if coordinates != expected_coordinates or len(tasks) != len(expected_coordinates):
        msg = "Release completion does not cover the exact 32x3 raw task grid."
        raise ValueError(msg)

    raw_tasks: dict[tuple[str, str], dict[str, Any]] = {}
    for task in tasks:
        cohort = str(task["cohort"])
        provider_key = str(task["provider"])
        task_name = f"provenance/run/tasks/{cohort}/{provider_key}/task_manifest.json"
        _require_record_path(
            task.get("manifest"),
            records,
            task_name,
            expected_path=f"tasks/{cohort}/{provider_key}/task_manifest.json",
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
    _require_record_path(
        raw_chain.get("input_manifest"),
        records,
        input_name,
        expected_path="input_manifest.json",
        label="attestation input",
    )
    _require_record_path(
        raw_chain.get("provider_manifest"),
        records,
        provider_name,
        expected_path="provider_manifest.json",
        label="attestation provider",
    )
    _require_record_path(
        raw_chain.get("run_manifest"),
        records,
        run_name,
        expected_path="run_manifest.json",
        label="attestation run",
    )
    _require_record_path(
        raw_chain.get("completion_manifest"),
        records,
        completion_name,
        expected_path="completion_manifest.json",
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
        or any(
            not isinstance(record, dict)
            or set(record) != {"cohort", "provider", "manifest", "outputs"}
            for record in attested_tasks
        )
        or len(attested_task_records) != len(expected_coordinates)
        or set(attested_task_records) != expected_coordinates
    ):
        msg = "Fit attestation does not cover the exact raw task grid."
        raise ValueError(msg)
    for coordinate, task_record in attested_task_records.items():
        cohort, provider_key = coordinate
        task_name = f"provenance/run/tasks/{cohort}/{provider_key}/task_manifest.json"
        _require_record_path(
            task_record.get("manifest"),
            records,
            task_name,
            expected_path=f"tasks/{cohort}/{provider_key}/task_manifest.json",
            label=f"attestation task {cohort}/{provider_key}",
        )
        if task_record.get("outputs") != raw_tasks[coordinate].get("outputs"):
            msg = f"Attested raw task outputs changed: {cohort}/{provider_key}"
            raise ValueError(msg)
    contract_evidence_value = raw_chain.get("cohort_contracts")
    if not isinstance(contract_evidence_value, list) or any(
        not isinstance(record, dict)
        or set(record)
        != {"path", "bytes", "sha256", "canonical_sha256"}
        or not isinstance(record.get("bytes"), int)
        or isinstance(record.get("bytes"), bool)
        or int(record["bytes"]) <= 0
        or not _valid_hash(record.get("sha256"))
        or not _valid_hash(record.get("canonical_sha256"))
        for record in contract_evidence_value
    ):
        msg = "Fit attestation cohort contract records violate their exact schema."
        raise ValueError(msg)
    contract_evidence = {
        str(record.get("path", ""))
        .removeprefix("contracts/")
        .removesuffix(".json"): record
        for record in contract_evidence_value
    }
    if (
        len(contract_evidence) != len(TCGA_COHORTS)
        or list(contract_evidence) != list(TCGA_COHORTS)
    ):
        msg = "Fit attestation does not cover the exact cohort contract grid."
        raise ValueError(msg)
    contract_features: dict[str, tuple[str, ...]] = {}
    contract_sample_counts: dict[str, int] = {}
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
            or not isinstance(source_contract, dict)
            or set(source_contract) != {"bytes", "sha256", "canonical_sha256"}
            or not isinstance(source_contract.get("bytes"), int)
            or isinstance(source_contract.get("bytes"), bool)
            or int(source_contract["bytes"]) <= 0
            or not _valid_hash(source_contract.get("sha256"))
            or not _valid_hash(source_contract.get("canonical_sha256"))
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
        contract_features[cohort] = tuple(str(item) for item in projection["features"])
        sample_count = projection.get("samples", {}).get("count")
        if (
            not isinstance(sample_count, int)
            or isinstance(sample_count, bool)
            or sample_count <= 0
        ):
            msg = f"Public cohort contract has an invalid sample count: {cohort}"
            raise ValueError(msg)
        if input_records[cohort].get("sample_count") != sample_count:
            msg = f"Public cohort and canonical input sample counts differ: {cohort}"
            raise ValueError(msg)
        contract_sample_counts[cohort] = sample_count
        for provider_key in ("cbase", "dig", "mutsig"):
            _validate_archived_raw_task(
                raw_tasks[(cohort, provider_key)],
                cohort=cohort,
                provider=provider_key,
                analysis_config_sha256=str(analysis_config_sha256),
                contract_sha256=str(source_contract.get("canonical_sha256")),
                pair_count=int(core._pair_contract(contract_features[cohort])["row_count"]),  # noqa: SLF001
            )

    post_root_name = "results/postprocess/postprocess_manifest.json"
    post_root = _json_member(archive, post_root_name)
    _require_record_path(
        post_root.get("run_completion"),
        records,
        completion_name,
        expected_path="completion_manifest.json",
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
        set(post_root)
        != {
            "schema_version",
            "contract",
            "effective_p_policy",
            "probability_representation",
            "multiplicity",
            "run_completion",
            "cohorts",
            "cohort_count",
            "provider_family_count",
            "pair_count_per_provider",
            "reporting_threshold_selected",
            "cohort_manifests",
        }
        or post_root.get("schema_version") != postprocess.SCHEMA_VERSION
        or post_root.get("contract") != postprocess.ROOT_CONTRACT
        or post_root.get("effective_p_policy")
        != "chi-square-one-df-for-full-affine-rank-otherwise-p-one"
        or post_root.get("probability_representation")
        != postprocess.PROBABILITY_REPRESENTATION
        or post_root.get("multiplicity")
        != {
            "primary": "benjamini-yekutieli",
            "nominal_sensitivity": "benjamini-hochberg",
        }
        or post_root.get("cohorts") != list(TCGA_COHORTS)
        or post_root.get("cohort_count") != len(TCGA_COHORTS)
        or post_root.get("provider_family_count") != len(TCGA_COHORTS) * 3
        or post_root.get("reporting_threshold_selected") is not False
        or set(postprocess_records) != expected_postprocess_manifests
    ):
        msg = "Postprocess root does not cover the exact cohort manifest grid."
        raise ValueError(msg)
    validated_pair_count = 0
    expected_summary_rows: dict[str, Mapping[str, int | float | str]] = {}
    expected_overlap_rows: list[dict[str, int | float | str]] = []
    expected_top_frames: list[pd.DataFrame] = []
    cohort_diagnostics: dict[str, Mapping[str, Any]] = {}
    for cohort in TCGA_COHORTS:
        cohort_manifest_name = (
            f"results/postprocess/{cohort}/{postprocess.COHORT_MANIFEST_NAME}"
        )
        result_name = f"results/postprocess/{cohort}/{postprocess.RESULT_NAME}"
        relative_manifest = f"{cohort}/{postprocess.COHORT_MANIFEST_NAME}"
        _require_record_path(
            postprocess_records[relative_manifest],
            records,
            cohort_manifest_name,
            expected_path=relative_manifest,
            label=f"postprocess root to cohort {cohort}",
        )
        cohort_manifest = _json_member(archive, cohort_manifest_name)
        cohort_providers = cohort_manifest.get("providers")
        pair_count = cohort_manifest.get("pair_count")
        if (
            set(cohort_manifest)
            != {
                "schema_version",
                "contract",
                "cohort",
                "pair_count",
                "providers",
                "family",
                "multiplicity",
                "direction",
                "non_full_rank",
                "probability_representation",
                "diagnostics",
                "reporting_threshold_selected",
                "sources",
                "output",
            }
            or cohort_manifest.get("schema_version") != postprocess.SCHEMA_VERSION
            or cohort_manifest.get("contract") != postprocess.DERIVATION_CONTRACT
            or cohort_manifest.get("cohort") != cohort
            or not isinstance(pair_count, int)
            or isinstance(pair_count, bool)
            or pair_count < 0
            or cohort_providers != ["cbase", "dig", "mutsig"]
            or cohort_manifest.get("family")
            != "all-matched-unordered-pairs-excluding-same-base-M:N"
            or cohort_manifest.get("multiplicity")
            != {
                "primary": (
                    "provider-specific-BY-over-complete-within-cohort-family"
                ),
                "nominal_sensitivity": (
                    "provider-specific-BH-over-complete-within-cohort-family"
                ),
            }
            or cohort_manifest.get("direction")
            != "rho-sign-after-nondirectional-profile-LRT"
            or cohort_manifest.get("non_full_rank")
            != "retain-in-family-with-p-one-and-no-directional-effect"
            or cohort_manifest.get("probability_representation")
            != postprocess.PROBABILITY_REPRESENTATION
            or not postprocess._valid_diagnostics(  # noqa: SLF001
                cohort_manifest.get("diagnostics"),
                pair_count,
            )
            or cohort_manifest.get("reporting_threshold_selected") is not False
        ):
            msg = f"Postprocess cohort manifest contract is invalid: {cohort}"
            raise ValueError(msg)
        _require_record_path(
            cohort_manifest.get("output"),
            records,
            result_name,
            expected_path=f"{cohort}/{postprocess.RESULT_NAME}",
            label=f"postprocess output {cohort}",
        )
        result_bytes = _verified_member_bytes(
            archive,
            result_name,
            records,
            limit=_INFERENCE_LIMIT_BYTES,
        )
        frame = pd.read_csv(BytesIO(result_bytes), float_precision="round_trip")
        _validate_archived_inference_frame(
            frame,
            cohort=cohort,
            features=contract_features[cohort],
        )
        if len(frame) != pair_count:
            msg = f"Archived inference row count differs from its receipt: {cohort}"
            raise ValueError(msg)
        expected_diagnostics = {}
        for provider_key in core.BMRS:
            effects = frame[f"{provider_key}_effect_identifiability"].astype("string")
            expected_diagnostics[provider_key] = {
                "full_affine_rank_count": int(effects.eq("full-affine-rank").sum()),
                "rank_deficient_count": int(effects.eq("rank-deficient").sum()),
                "rank_not_certified_underflow_count": int(
                    effects.eq("rank-not-certified-underflow").sum(),
                ),
                "p_display_clipped_count": int(
                    (
                        frame[f"{provider_key}_log_p_value"].to_numpy(dtype=float)
                        < postprocess.LOG_MIN_POSITIVE_FLOAT
                    ).sum(),
                ),
                "by_display_clipped_count": int(
                    (
                        frame[f"{provider_key}_log_by_q_value"].to_numpy(dtype=float)
                        < postprocess.LOG_MIN_POSITIVE_FLOAT
                    ).sum(),
                ),
                "bh_display_clipped_count": int(
                    (
                        frame[f"{provider_key}_log_bh_q_value"].to_numpy(dtype=float)
                        < postprocess.LOG_MIN_POSITIVE_FLOAT
                    ).sum(),
                ),
            }
        if cohort_manifest["diagnostics"] != expected_diagnostics:
            msg = f"Postprocess diagnostics differ from inference: {cohort}"
            raise ValueError(msg)
        expected_summary_rows[cohort] = _expected_summary_decisions(
            cohort=cohort,
            frame=frame,
            features=contract_features[cohort],
            sample_count=contract_sample_counts[cohort],
        )
        expected_overlap_rows.extend(
            reporting._overlap_rows(  # noqa: SLF001
                frame,
                cohort=cohort,
                primary_adjustment="benjamini-yekutieli",
                primary_q=0.01,
            ),
        )
        expected_top_frames.append(
            reporting._top_primary_pairs(  # noqa: SLF001
                frame,
                cohort=cohort,
                primary_adjustment="benjamini-yekutieli",
                primary_q=0.01,
            ),
        )
        cohort_diagnostics[cohort] = cohort_manifest["diagnostics"]
        validated_pair_count += pair_count
        for provider_key in ("cbase", "dig", "mutsig"):
            if raw_tasks[(cohort, provider_key)].get("pairwise_rows") != pair_count:
                msg = (
                    "Raw and postprocessed pair counts differ: "
                    f"{cohort}/{provider_key}"
                )
                raise ValueError(msg)
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
    if post_root.get("pair_count_per_provider") != validated_pair_count:
        msg = "Postprocess root pair count differs from validated cohort families."
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
    _require_record_path(
        calibration_run.get("run_completion"),
        records,
        completion_name,
        expected_path="completion_manifest.json",
        label="calibration to completion",
    )
    _require_record_path(
        calibration_run.get("provider_manifest"),
        records,
        provider_name,
        expected_path="provider_manifest.json",
        label="calibration to provider",
    )
    _require_record_path(
        calibration_summary.get("run_manifest"),
        records,
        calibration_run_name,
        expected_path="run_manifest.json",
        label="calibration summary to run",
    )
    _require_record_path(
        calibration_summary.get("table"),
        records,
        calibration_table_name,
        expected_path=calibration.SUMMARY_TABLE_NAME,
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
        _require_record_path(
            task.get("output"),
            records,
            data_name,
            expected_path=calibration.TASK_DATA_NAME,
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
    report_inputs = report.get("inputs")
    high_burden = report.get("high_burden_definition")
    if (
        set(report)
        != {
            "schema_version",
            "contract",
            "cohorts",
            "primary_provider",
            "inference_status",
            "effective_p_policy",
            "primary_adjustment",
            "primary_q_threshold",
            "sensitivity_adjustment",
            "sensitivity_q_threshold",
            "provider_overlap",
            "threshold_decision_scale",
            "probability_representation",
            "sample_level_rows_included",
            "burden_source_policy",
            "high_burden_definition",
            "inputs",
            "outputs",
        }
        or report.get("schema_version") != reporting.SCHEMA_VERSION
        or report.get("contract") != reporting.REPORT_CONTRACT
        or report.get("cohorts") != list(TCGA_COHORTS)
        or report.get("primary_provider") != "mutsig"
        or report.get("inference_status") != rule_module.REPORTABLE_STATUS
        or report.get("effective_p_policy")
        != "chi-square-one-df-for-full-affine-rank-otherwise-p-one"
        or report.get("primary_adjustment") != "benjamini-yekutieli"
        or report.get("primary_q_threshold") != 0.01
        or report.get("sensitivity_adjustment") != "benjamini-hochberg"
        or report.get("sensitivity_q_threshold") != 0.01
        or report.get("provider_overlap")
        != "direction-concordant-descriptive-only-not-an-inferential-vote"
        or report.get("threshold_decision_scale") != "natural-log-q-values"
        or report.get("probability_representation")
        != postprocess.PROBABILITY_REPRESENTATION
        or report.get("sample_level_rows_included") is not False
        or report.get("burden_source_policy")
        != reporting.BURDEN_SOURCE_POLICY
        or not isinstance(report_inputs, dict)
        or set(report_inputs)
        != {
            "run_completion",
            "provider_manifest",
            "postprocess_manifest",
            "calibration_summary",
            "reporting_rule",
        }
        or not isinstance(high_burden, dict)
        or set(high_burden)
        != {
            "measure",
            "reference",
            "pooled_tumor_count",
            "quantile",
            "threshold",
            "source",
            "comparison",
            "interpretation",
        }
        or high_burden.get("measure")
        != "pre-K total nonsynonymous SNV event count per tumor"
        or high_burden.get("reference")
        != "pooled 10,433-tumor 32-cohort analysis population"
        or high_burden.get("pooled_tumor_count") != reporting.EXPECTED_TUMOR_COUNT
        or high_burden.get("quantile") != reporting.HIGH_BURDEN_QUANTILE
        or high_burden.get("source") != "cohort_burden_histogram.csv"
        or not isinstance(high_burden.get("threshold"), (int, float))
        or isinstance(high_burden.get("threshold"), bool)
        or not np.isfinite(float(high_burden["threshold"]))
        or float(high_burden["threshold"]) < 0
        or high_burden.get("comparison") != "greater-than-or-equal"
        or high_burden.get("interpretation")
        != "descriptive high-burden fraction, not a clinical hypermutator label"
    ):
        msg = "Archived report manifest violates the frozen reporting contract."
        raise ValueError(msg)
    for key, member_name, expected_path in (
        ("run_completion", completion_name, "completion_manifest.json"),
        ("provider_manifest", provider_name, "provider_manifest.json"),
        ("postprocess_manifest", post_root_name, postprocess.ROOT_MANIFEST_NAME),
        (
            "calibration_summary",
            calibration_summary_name,
            calibration.SUMMARY_NAME,
        ),
        ("reporting_rule", rule_name, "reporting_rule.json"),
    ):
        _require_record_path(
            report_inputs.get(key),
            records,
            member_name,
            expected_path=expected_path,
            label=f"report input {key}",
        )
    _require_record_path(
        report.get("inputs", {}).get("reporting_rule"),
        records,
        rule_name,
        expected_path="reporting_rule.json",
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
        _require_record_path(
            output_record,
            records,
            f"results/report/{output_name}",
            expected_path=output_name,
            label=f"report output {output_name}",
        )
    _validate_archived_report_derivations(
        archive,
        records,
        expected_summary_rows=expected_summary_rows,
        expected_overlap=pd.DataFrame(
            expected_overlap_rows,
            columns=reporting.OVERLAP_COLUMNS,
        ),
        expected_top=pd.concat(expected_top_frames, ignore_index=True),
        sample_counts=contract_sample_counts,
        raw_tasks=raw_tasks,
        cohort_diagnostics=cohort_diagnostics,
        calibration_table_name=calibration_table_name,
        high_burden_threshold=float(high_burden["threshold"]),
    )

    document_manifest_name = f"documents/{DOCUMENT_MANIFEST_NAME}"
    document_manifest = _json_member(archive, document_manifest_name)
    _require_record_path(
        document_manifest.get("inputs", {}).get("report_manifest"),
        records,
        report_manifest_name,
        expected_path="report_manifest.json",
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
        set(document_manifest)
        != {"schema_version", "contract", "inputs", "outputs"}
        or document_manifest.get("schema_version") != SCHEMA_VERSION
        or document_manifest.get("contract") != DOCUMENT_CONTRACT
        or not isinstance(document_manifest.get("inputs"), dict)
        or set(document_manifest["inputs"]) != {"report_manifest"}
        or set(document_outputs) != REQUIRED_DOCUMENTS
        or archived_document_members != expected_document_members
    ):
        msg = "Document manifest does not list the exact submission documents."
        raise ValueError(msg)
    for name, record in document_outputs.items():
        _require_record_path(
            record,
            records,
            f"documents/{name}",
            expected_path=name,
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
    with path.open("rb") as compressed:
        if compressed.read(10) != bytes.fromhex("1f8b08000000000002ff"):
            msg = "Release gzip header is not the canonical deterministic wrapper."
            raise ValueError(msg)
    with tarfile.open(path, mode="r:gz") as archive:
        infos = archive.getmembers()
        names = [info.name for info in infos]
        if names != sorted(names) or len(names) != len(set(names)):
            msg = "Release archive member order or uniqueness is invalid."
            raise ValueError(msg)
        if archive.pax_headers or any(
            not info.isfile()
            or info.mode != 0o444
            or info.mtime != 0
            or info.uid != 0
            or info.gid != 0
            or info.uname != ""
            or info.gname != ""
            or info.linkname != ""
            or info.pax_headers
            or info.devmajor != 0
            or info.devminor != 0
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
        exact_privacy = {
            "raw_tumor_level_inputs": "excluded",
            "sample_identifiers": "excluded",
            "sample_specific_mutsig_tensor": "excluded",
            "restricted_mutsig_source": "excluded",
        }
        if (
            set(manifest)
            != {
                "schema_version",
                "contract",
                "fit_source_commit",
                "release_source_commit",
                "member_count",
                "members",
                "privacy",
            }
            or manifest.get("schema_version") != SCHEMA_VERSION
            or manifest.get("contract") != RELEASE_CONTRACT
            or manifest.get("member_count") != len(records)
            or len(records) != len(manifest_records)
            or set(records) != payload_names
            or manifest.get("privacy") != exact_privacy
            or manifest.get("fit_source_commit") != provenance.PRODUCTION_FIT_COMMIT
            or re.fullmatch(
                r"[0-9a-f]{40}",
                str(manifest.get("release_source_commit", "")),
            )
            is None
            or any(
                set(record) != {"path", "bytes", "sha256"}
                or record.get("path") != name
                or not isinstance(record.get("bytes"), int)
                or isinstance(record.get("bytes"), bool)
                or int(record["bytes"]) < 0
                or re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256", "")))
                is None
                for name, record in records.items()
            )
        ):
            msg = "Release manifest does not cover the exact payload."
            raise ValueError(msg)
        _require_embedded_reportable_gate(archive, records)
        _recompute_archived_calibration_gate(archive, records)
        _verify_canonical_archive_stream(path, infos)
        for name in sorted(payload_names):
            handle = archive.extractfile(name)
            if handle is None:
                msg = f"Release member cannot be read: {name}"
                raise ValueError(msg)
            size, digest = _hash_stream(handle, public_name=name)
            if size != records[name]["bytes"] or digest != records[name]["sha256"]:
                msg = f"Release member digest differs: {name}"
                raise ValueError(msg)
            _assert_aggregate_csv_header(archive, name)
            if PurePosixPath(name).suffix.casefold() == ".pdf":
                _scan_pdf_privacy(
                    _verified_member_bytes(
                        archive,
                        name,
                        records,
                        limit=_PDF_LIMIT_BYTES,
                    ),
                    name=name,
                )
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
    rule = _load_strict_json_path(
        rule_path,
        public_name=REPORTING_RULE_MEMBER,
    )
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
    receipt = _load_strict_json_path(
        receipt_path,
        public_name=receipt_path.name,
    )
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
