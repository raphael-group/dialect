"""Build and independently verify the focused DIALECT submission release.

The archive contains the final documents, complete provider-specific derived
association tables, calibration data, reporting artifacts, and execution receipts.
It deliberately excludes raw tumor-level inputs and the restricted MutSig source.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import tarfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Final

from analysis import calibrate_tcga_revision_focused as calibration
from analysis import postprocess_tcga_revision_focused as postprocess
from analysis import report_tcga_revision_focused as reporting
from dialect.data.tcga import TCGA_COHORTS

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import BinaryIO

SCHEMA_VERSION: Final = "1.0.0"
RELEASE_CONTRACT: Final = "focused-dialect-submission-release-v1"
RECEIPT_CONTRACT: Final = "focused-dialect-submission-release-receipt-v1"
MANIFEST_NAME: Final = "release_manifest.json"
SOURCE_RECORD_NAME: Final = "provenance/source_commit.json"
README_NAME: Final = "README.md"
REQUIRED_DOCUMENTS: Final = {
    "manuscript.tex",
    "manuscript.pdf",
    "marked_manuscript.pdf",
    "response_to_reviewers.pdf",
    "supporting_information.tex",
    "supporting_information.pdf",
    "rebuttal.md",
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


def _file_member(name: str, path: Path) -> Member:
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
    return Member(
        name=_safe_name(name),
        path=None,
        content=content,
        size=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )


def _document_members(document_root: Path) -> list[Member]:
    files = sorted(path for path in document_root.rglob("*") if path.is_file())
    if any(path.is_symlink() for path in document_root.rglob("*")):
        msg = "Submission document root may not contain symlinks."
        raise ValueError(msg)
    relative_names = {path.relative_to(document_root).as_posix() for path in files}
    if relative_names < REQUIRED_DOCUMENTS:
        missing = sorted(REQUIRED_DOCUMENTS - relative_names)
        msg = f"Submission document root is missing required files: {missing}"
        raise ValueError(msg)
    return [
        _file_member(
            f"documents/{path.relative_to(document_root).as_posix()}",
            path,
        )
        for path in files
    ]


def _result_members(
    *,
    run_root: Path,
    postprocess_root: Path,
    calibration_root: Path,
    report_root: Path,
    rule_path: Path,
) -> list[Member]:
    postprocess.validate_derived_root(postprocess_root, TCGA_COHORTS)
    calibration.validate_summary(calibration_root)
    report_manifest = reporting.validate_report(report_root)
    rule_record = report_manifest.get("inputs", {}).get("reporting_rule", {})
    if (
        rule_record.get("bytes") != rule_path.stat().st_size
        or rule_record.get("sha256") != _sha256_path(rule_path)
    ):
        msg = "Reporting rule differs from the rule used to build final artifacts."
        raise ValueError(msg)
    members = [
        _file_member(
            "results/postprocess/postprocess_manifest.json",
            postprocess_root / postprocess.ROOT_MANIFEST_NAME,
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
        _file_member(
            "provenance/completion_manifest.json",
            run_root / "completion_manifest.json",
        ),
    ]
    for cohort in TCGA_COHORTS:
        members.extend(
            [
                _file_member(
                    f"results/postprocess/{cohort}/{name}",
                    postprocess_root / cohort / name,
                )
                for name in (
                    postprocess.RESULT_NAME,
                    postprocess.COHORT_MANIFEST_NAME,
                )
            ],
        )
        members.extend(
            [
                _file_member(
                    f"provenance/tasks/{cohort}/{provider}/task_manifest.json",
                    run_root / "tasks" / cohort / provider / "task_manifest.json",
                )
                for provider in ("cbase", "dig", "mutsig")
            ],
        )
    members.extend(
        _file_member(f"results/report/{path.name}", path)
        for path in sorted(report_root.iterdir())
        if path.is_file()
    )
    for cohort in calibration._load_config()["cells"]["cohorts"]:  # noqa: SLF001
        for provider in ("cbase", "dig", "mutsig"):
            task_root = calibration_root / "tasks" / cohort / provider
            members.extend(
                [
                    _file_member(
                        f"results/calibration/tasks/{cohort}/{provider}/{name}",
                        task_root / name,
                    )
                    for name in (
                        calibration.TASK_DATA_NAME,
                        calibration.TASK_MANIFEST_NAME,
                    )
                ],
            )
    return members


def _source_record(repository_root: Path, expected_commit: str) -> Member:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    observed = completed.stdout.strip()
    if observed != expected_commit:
        msg = (
            f"Repository HEAD {observed} differs from release commit "
            f"{expected_commit}."
        )
        raise ValueError(msg)
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        msg = "Tracked repository files must be clean for release."
        raise ValueError(msg)
    remote = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    record = {
        "commit": observed,
        "remote": remote,
        "branch": "codex/revision-focused",
        "restricted_mutsig_source_included": False,
        "raw_tumor_level_inputs_included": False,
    }
    return _bytes_member(SOURCE_RECORD_NAME, _canonical_json(record) + b"\n")


def _readme(commit: str) -> Member:
    content = (
        "# DIALECT focused revision release\n\n"
        "This immutable candidate accompanies the corrected 32-cohort K=500 analysis.\n"
        "MutSig is the primary background for ME and CO; CBaSE is retained for ME "
        "continuity and DIG as supplementary sensitivity. Provider overlap is "
        "descriptive, not an inferential vote.\n\n"
        f"Source commit: `{commit}`.\n\n"
        "The archive excludes raw tumor-level inputs and the restricted MutSig source. "
        "Derived association tables contain mutation-event pairs and statistics only.\n"
    ).encode()
    return _bytes_member(README_NAME, content)


def _manifest(members: Sequence[Member], source_commit: str) -> bytes:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "contract": RELEASE_CONTRACT,
        "source_commit": source_commit,
        "member_count": len(members),
        "members": [
            {"path": member.name, "bytes": member.size, "sha256": member.sha256}
            for member in members
        ],
        "privacy": {
            "raw_tumor_level_inputs": "excluded",
            "sample_identifiers": "excluded",
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
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw,
            mtime=0,
        ) as compressed, tarfile.open(
            fileobj=compressed,
            mode="w",
            format=tarfile.USTAR_FORMAT,
        ) as archive:
            ordered = sorted(
                (*members, manifest_member),
                key=lambda item: item.name,
            )
            for member in ordered:
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


def verify_archive(path: Path) -> dict[str, Any]:
    """Independently verify every payload byte against the internal manifest."""
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
        manifest_handle = archive.extractfile(MANIFEST_NAME)
        if manifest_handle is None:
            msg = "Release manifest is missing."
            raise ValueError(msg)
        manifest = json.loads(manifest_handle.read().decode("utf-8"))
        records = {record["path"]: record for record in manifest.get("members", [])}
        payload_names = set(names) - {MANIFEST_NAME}
        if (
            manifest.get("schema_version") != SCHEMA_VERSION
            or manifest.get("contract") != RELEASE_CONTRACT
            or manifest.get("member_count") != len(records)
            or set(records) != payload_names
        ):
            msg = "Release manifest does not cover the exact payload."
            raise ValueError(msg)
        for name in sorted(payload_names):
            handle = archive.extractfile(name)
            if handle is None:
                msg = f"Release member cannot be read: {name}"
                raise ValueError(msg)
            size, digest = _hash_stream(handle)
            if size != records[name]["bytes"] or digest != records[name]["sha256"]:
                msg = f"Release member digest differs: {name}"
                raise ValueError(msg)
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
    source_commit: str,
    run_root: Path,
    postprocess_root: Path,
    calibration_root: Path,
    report_root: Path,
    rule_path: Path,
    document_root: Path,
    destination: Path,
    receipt_path: Path,
) -> Path:
    """Build, verify, and atomically publish one immutable release candidate."""
    if destination.exists() or receipt_path.exists():
        msg = "Release archive and receipt destinations must both be unused."
        raise FileExistsError(msg)
    members = [
        *_document_members(document_root),
        *_result_members(
            run_root=run_root,
            postprocess_root=postprocess_root,
            calibration_root=calibration_root,
            report_root=report_root,
            rule_path=rule_path,
        ),
        _source_record(repository_root, source_commit),
        _readme(source_commit),
    ]
    names = [member.name for member in members]
    if len(names) != len(set(names)):
        msg = "Release plan contains duplicate member names."
        raise ValueError(msg)
    manifest = _manifest(sorted(members, key=lambda item: item.name), source_commit)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        _write_archive(staging, members, manifest)
        verified = verify_archive(staging)
        archive_sha256 = _sha256_path(staging)
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "contract": RECEIPT_CONTRACT,
            "archive": {
                "path": destination.name,
                "bytes": staging.stat().st_size,
                "sha256": archive_sha256,
            },
            "release_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
            "source_commit": source_commit,
            "member_count": verified["member_count"],
        }
        receipt_raw = _canonical_json(receipt) + b"\n"
        receipt_staging = receipt_path.with_name(
            f".{receipt_path.name}.{os.getpid()}.tmp",
        )
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        with receipt_staging.open("xb") as handle:
            handle.write(receipt_raw)
            handle.flush()
            os.fsync(handle.fileno())
        _publish_no_replace(staging, destination)
        _publish_no_replace(receipt_staging, receipt_path)
    finally:
        staging.unlink(missing_ok=True)
        receipt_path.with_name(f".{receipt_path.name}.{os.getpid()}.tmp").unlink(
            missing_ok=True,
        )
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
        or receipt.get("source_commit") != manifest.get("source_commit")
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
    parser.add_argument("--source-commit")
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--postprocess-root", type=Path)
    parser.add_argument("--calibration-root", type=Path)
    parser.add_argument("--report-root", type=Path)
    parser.add_argument("--reporting-rule", type=Path)
    parser.add_argument("--document-root", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def _required_build_args(args: argparse.Namespace) -> Mapping[str, object]:
    names = (
        "repository_root",
        "source_commit",
        "run_root",
        "postprocess_root",
        "calibration_root",
        "report_root",
        "reporting_rule",
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
            source_commit=str(values["source_commit"]),
            run_root=Path(values["run_root"]).resolve(),
            postprocess_root=Path(values["postprocess_root"]).resolve(),
            calibration_root=Path(values["calibration_root"]).resolve(),
            report_root=Path(values["report_root"]).resolve(),
            rule_path=Path(values["reporting_rule"]).resolve(),
            document_root=Path(values["document_root"]).resolve(),
            destination=Path(values["output"]).absolute(),
            receipt_path=Path(values["receipt"]).absolute(),
        ),
    )


if __name__ == "__main__":
    main()
