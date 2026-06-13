"""Download TCGA PanCancer Atlas MAFs for every DIALECT cohort from cBioPortal.

Pulls ``data_mutations.txt`` for each cohort's ``<code>_tcga_pan_can_atlas_2018``
study from the cBioPortal datahub (git-lfs media endpoint, which serves the raw
file), writing ``data/mafs_pancan/<COHORT>.maf``. Using one uniform source for all
cohorts keeps the cross-cohort comparison clean. Idempotent: skips a cohort whose
MAF already exists and has a valid ``Hugo_Symbol`` header.

Usage::

    python scripts/download_cbioportal_mafs.py            # all cohorts
    python scripts/download_cbioportal_mafs.py ACC BLCA   # a subset
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

DOWNLOADS = Path.home() / "Downloads"

# The paper's cohort set (= data/event_level_likely_passengers/*.txt).
COHORTS = [
    "ACC", "BLCA", "BRCA", "CESC", "CHOL", "CRAD", "DLBC", "ESCA", "GBM", "HNSC",
    "KICH", "KIRC", "KIRP", "LAML", "LGG", "LIHC", "LUAD", "LUSC", "MESO", "OV",
    "PAAD", "PCPG", "PRAD", "SARC", "SKCM", "STAD", "TGCT", "THCA", "THYM", "UCEC",
    "UCS", "UVM",
]
# cBioPortal study code differs from the DIALECT cohort code for these:
STUDY_CODE_OVERRIDE = {"CRAD": "coadread"}
BASE = (
    "https://media.githubusercontent.com/media/cBioPortal/datahub/master/public/"
    "{study}_tcga_pan_can_atlas_2018/data_mutations.txt"
)
OUT_DIR = Path("data/mafs_pancan")
MIN_MAF_BYTES = 1000


def study_url(cohort: str) -> str:
    """Datahub URL for a cohort's PanCancer Atlas mutations file (cBioPortal)."""
    code = STUDY_CODE_OVERRIDE.get(cohort, cohort.lower())
    return BASE.format(study=code)


def is_valid_maf(path: Path) -> bool:
    """True if the file exists and starts with a Hugo_Symbol-led MAF header."""
    if not path.exists() or path.stat().st_size < MIN_MAF_BYTES:
        return False
    with path.open() as f:
        return f.readline().startswith("Hugo_Symbol")


def from_local_tarball(cohort: str, out: Path) -> bool:
    """Extract data_mutations.txt from a cohort's tarball in ~/Downloads, if present.

    Matches browser duplicate-download names too (e.g. ``luad_..._2018 (1).tar.gz``);
    the member path inside is always the un-suffixed study directory.
    """
    code = STUDY_CODE_OVERRIDE.get(cohort, cohort.lower())
    study = f"{code}_tcga_pan_can_atlas_2018"
    candidates = sorted(DOWNLOADS.glob(f"{study}*.tar.gz"))
    member = f"{study}/data_mutations.txt"
    for tarball in candidates:
        with out.open("wb") as fh:
            proc = subprocess.run(
                ["tar", "-xzf", str(tarball), "-O", member],
                stdout=fh,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        if proc.returncode == 0 and is_valid_maf(out):
            return True
    out.unlink(missing_ok=True)
    return False


def download(cohort: str) -> tuple[str, str]:
    """Get one cohort's MAF: reuse, then local tarball, then datahub; return status."""
    out = OUT_DIR / f"{cohort}.maf"
    if is_valid_maf(out):
        return cohort, f"skip (exists, {out.stat().st_size // 1_000_000} MB)"
    src = "tarball" if from_local_tarball(cohort, out) else None
    if src is None:
        proc = subprocess.run(
            [
                "curl", "-sf", "--retry", "6", "--retry-delay", "5",
                "--retry-all-errors", "-m", "1200",
                study_url(cohort), "-o", str(out),
            ],
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            out.unlink(missing_ok=True)
            return cohort, f"FAIL (curl {proc.returncode}) {study_url(cohort)}"
        if not is_valid_maf(out):
            out.unlink(missing_ok=True)
            return cohort, "FAIL (bad header from datahub)"
        src = "datahub"
    n = sum(1 for _ in out.open()) - 1
    return cohort, f"ok via {src} ({out.stat().st_size // 1_000_000} MB, {n} muts)"


def main() -> None:
    """Download MAFs for the requested cohorts (default: all)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cohorts = [c.upper() for c in sys.argv[1:]] or COHORTS
    failures = []
    for cohort in cohorts:
        name, status = download(cohort)
        print(f"{name:6} {status}", flush=True)
        if status.startswith("FAIL"):
            failures.append(name)
    print(f"\n{len(cohorts) - len(failures)}/{len(cohorts)} ok"
          + (f"; FAILED: {failures}" if failures else ""))


if __name__ == "__main__":
    main()
