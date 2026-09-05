"""Build and verify the focused submission document manifest.

The document manifest binds the exact submission-facing source and rendered
documents to the validated scientific report manifest.  It is intentionally
small and fail-closed: no extra files, symlinks, or in-place replacement are
accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Final

from analysis import build_tcga_revision_focused_release as release

if TYPE_CHECKING:
    from collections.abc import Sequence

_SHA256_LENGTH: Final = 64


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


def _record(path: Path, *, public_path: str) -> dict[str, int | str]:
    if not path.is_file() or path.is_symlink():
        msg = f"Required document-manifest input is missing or unsafe: {path}"
        raise ValueError(msg)
    return {
        "path": public_path,
        "bytes": path.stat().st_size,
        "sha256": _sha256_path(path),
    }


def _expected_document_paths(document_root: Path) -> dict[str, Path]:
    if (
        not document_root.is_dir()
        or document_root.is_symlink()
        or any(path.is_symlink() for path in document_root.rglob("*"))
    ):
        msg = "Submission document root must be a real directory without symlinks."
        raise ValueError(msg)
    observed = {
        path.relative_to(document_root).as_posix()
        for path in document_root.rglob("*")
        if path.is_file()
    }
    allowed = {*release.REQUIRED_DOCUMENTS}
    manifest_name = release.DOCUMENT_MANIFEST_NAME
    if manifest_name in observed:
        allowed.add(manifest_name)
    if observed != allowed:
        msg = "Submission document root must contain exactly the required files."
        raise ValueError(msg)
    return {
        name: document_root / name for name in sorted(release.REQUIRED_DOCUMENTS)
    }


def expected_manifest(
    *,
    document_root: Path,
    report_manifest: Path,
) -> dict[str, object]:
    """Return the canonical manifest for an exact submission document set."""
    documents = _expected_document_paths(document_root)
    if report_manifest.name != "report_manifest.json":
        msg = "The bound report input must be named report_manifest.json."
        raise ValueError(msg)
    return {
        "schema_version": release.SCHEMA_VERSION,
        "contract": release.DOCUMENT_CONTRACT,
        "inputs": {
            "report_manifest": _record(
                report_manifest,
                public_path="report_manifest.json",
            ),
        },
        "outputs": {
            name: _record(path, public_path=name)
            for name, path in documents.items()
        },
    }


def _write_once(path: Path, content: bytes) -> None:
    if path.exists() or path.is_symlink():
        msg = f"Refusing to overwrite document manifest: {path}"
        raise FileExistsError(msg)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def build_manifest(*, document_root: Path, report_manifest: Path) -> Path:
    """Create the immutable canonical document manifest and return its path."""
    output = document_root / release.DOCUMENT_MANIFEST_NAME
    if output.exists() or output.is_symlink():
        msg = f"Refusing to overwrite document manifest: {output}"
        raise FileExistsError(msg)
    manifest = expected_manifest(
        document_root=document_root,
        report_manifest=report_manifest,
    )
    _write_once(output, _canonical_json(manifest) + b"\n")
    verify_manifest(document_root=document_root, report_manifest=report_manifest)
    return output


def verify_manifest(*, document_root: Path, report_manifest: Path) -> Path:
    """Verify the canonical manifest and all bound files without mutation."""
    output = document_root / release.DOCUMENT_MANIFEST_NAME
    if not output.is_file() or output.is_symlink():
        msg = f"Document manifest is missing or unsafe: {output}"
        raise ValueError(msg)
    expected = expected_manifest(
        document_root=document_root,
        report_manifest=report_manifest,
    )
    expected_bytes = _canonical_json(expected) + b"\n"
    if output.read_bytes() != expected_bytes:
        msg = "Document manifest is noncanonical or no longer matches its inputs."
        raise ValueError(msg)
    release._document_members(document_root)  # noqa: SLF001
    digest = _sha256_path(output)
    if len(digest) != _SHA256_LENGTH:
        msg = "Unexpected document-manifest digest length."
        raise AssertionError(msg)
    return output


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document-root", required=True, type=Path)
    parser.add_argument("--report-manifest", required=True, type=Path)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify an existing manifest instead of creating one.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Build or verify the manifest from the command line."""
    args = _parse_args(argv)
    operation = verify_manifest if args.verify else build_manifest
    output = operation(
        document_root=args.document_root,
        report_manifest=args.report_manifest,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
