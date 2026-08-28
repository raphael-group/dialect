"""Download TCGA PanCancer Atlas MAFs for every DIALECT cohort from cBioPortal.

Pulls ``data_mutations.txt`` for every frozen TCGA study from the exact cBioPortal
DataHub revision commit (the Git-LFS media endpoint serves the raw file), writing
``data/mafs_pancan/<COHORT>.maf``. Each file must match its frozen SHA-256 receipt.
Existing files that do not match are refused rather than silently overwritten.

Usage::

    python scripts/download_cbioportal_mafs.py            # all cohorts
    python scripts/download_cbioportal_mafs.py ACC BLCA   # a subset
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from dialect.data.tcga import (
    TCGA_COHORTS,
    TCGA_DATAHUB_COMMIT,
    TCGA_MAF_SHA256,
    tcga_datahub_public_path,
)

BASE = (
    "https://media.githubusercontent.com/media/cBioPortal/datahub/"
    f"{TCGA_DATAHUB_COMMIT}/{{path}}"
)
OUT_DIR = Path("data/mafs_pancan")
MIN_MAF_BYTES = 1000


def study_url(cohort: str) -> str:
    """Return the immutable DataHub URL for one cohort mutation file."""
    path = tcga_datahub_public_path(cohort, "data_mutations.txt")
    return BASE.format(path=path.as_posix())


def _sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest without loading a MAF into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def maf_validation_error(path: Path, cohort: str) -> str | None:
    """Return why a MAF violates the frozen receipt, or ``None`` if valid."""
    if not path.is_file():
        return "missing file"
    if path.stat().st_size < MIN_MAF_BYTES:
        return f"file is smaller than {MIN_MAF_BYTES} bytes"
    try:
        with path.open(encoding="utf-8") as handle:
            if not handle.readline().startswith("Hugo_Symbol"):
                return "first row is not a Hugo_Symbol-led MAF header"
    except UnicodeDecodeError:
        return "file is not valid UTF-8"
    observed = _sha256(path)
    expected = TCGA_MAF_SHA256[cohort]
    if observed != expected:
        return f"SHA-256 mismatch (expected {expected}, observed {observed})"
    return None


def download(cohort: str) -> tuple[str, str]:
    """Download and atomically install one byte-attested cohort MAF."""
    out = OUT_DIR / f"{cohort}.maf"
    if os.path.lexists(out):
        error = maf_validation_error(out, cohort)
        if error is None:
            return cohort, f"skip (pinned, {out.stat().st_size // 1_000_000} MB)"
        return cohort, f"FAIL (existing file refused: {error})"

    with tempfile.NamedTemporaryFile(
        dir=out.parent,
        prefix=f".{cohort}.",
        suffix=".partial",
        delete=False,
    ) as handle:
        partial = Path(handle.name)
    try:
        proc = subprocess.run(
            [
                "curl",
                "-sf",
                "--retry",
                "6",
                "--retry-delay",
                "5",
                "--retry-all-errors",
                "-m",
                "1200",
                study_url(cohort),
                "-o",
                str(partial),
            ],
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            return cohort, f"FAIL (curl {proc.returncode}) {study_url(cohort)}"
        error = maf_validation_error(partial, cohort)
        if error is not None:
            return cohort, f"FAIL (download does not match pinned receipt: {error})"
        try:
            os.link(partial, out)
        except FileExistsError:
            return cohort, "FAIL (destination appeared during download; refused)"
    finally:
        partial.unlink(missing_ok=True)
    return cohort, f"ok (pinned, {out.stat().st_size // 1_000_000} MB)"


def main() -> None:
    """Download MAFs for the requested cohorts (default: all)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cohorts = [c.upper() for c in sys.argv[1:]] or list(TCGA_COHORTS)
    failures = []
    for cohort in cohorts:
        name, status = download(cohort)
        print(f"{name:6} {status}", flush=True)
        if status.startswith("FAIL"):
            failures.append(name)
    print(f"\n{len(cohorts) - len(failures)}/{len(cohorts)} ok"
          + (f"; FAILED: {failures}" if failures else ""))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
