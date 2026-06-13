"""Split an MSK panel MAF into per-cancer-type sub-cohort MAFs for DIALECT.

MSK-IMPACT/MSK-CHORD pool many cancer types in one cohort; running DIALECT on the pooled
conflates a cancer-type-composition confound with real CO. This splits the MAF by the
clinical ``CANCER_TYPE`` (broad level; ONCOTREE_CODE is far too granular -- 502
codes for IMPACT) into one MAF per type with at least ``--min-samples`` mutated samples,
mirroring the TCGA per-cohort design.

Usage::

    python scripts/split_msk_by_cancer_type.py \
        --maf data/mafs_msk/MSK_IMPACT_2026.maf \
        --clinical data/mafs_msk/MSK_IMPACT_2026.clinical.txt \
        --out data/mafs_msk_split/IMPACT2026 --min-samples 100
"""

from __future__ import annotations

import argparse
import io
import re
from pathlib import Path

import pandas as pd


def slug(name: str) -> str:
    """Filesystem-safe cohort label from a cancer-type string."""
    s = re.sub(r"[^0-9A-Za-z]+", "_", str(name)).strip("_")
    return s or "UNKNOWN"


def load_sample_to_type(clinical: Path, field: str) -> dict:
    """Map SAMPLE_ID -> cancer type, skipping cBioPortal's leading '#' comment lines."""
    with clinical.open() as fh:
        lines = [ln for ln in fh if not ln.startswith("#")]
    df = pd.read_csv(io.StringIO("".join(lines)), sep="\t", dtype=str)
    return dict(zip(df["SAMPLE_ID"], df[field], strict=False))


def main() -> None:
    """Write one per-cancer-type MAF for each type with enough mutated samples."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--maf", type=Path, required=True)
    ap.add_argument("--clinical", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--field", default="CANCER_TYPE")
    ap.add_argument("--min-samples", type=int, default=100)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    sample_to_type = load_sample_to_type(args.clinical, args.field)
    maf = pd.read_csv(args.maf, sep="\t", low_memory=False, comment="#")
    matched = maf["Tumor_Sample_Barcode"].isin(sample_to_type)
    print(f"MAF rows: {len(maf)}; sample-barcode match rate: {matched.mean():.1%}")
    maf = maf[matched].copy()
    maf["_TYPE"] = maf["Tumor_Sample_Barcode"].map(sample_to_type)

    kept, dropped = [], []
    for ctype, grp in maf.groupby("_TYPE"):
        n_samp = grp["Tumor_Sample_Barcode"].nunique()
        label = slug(ctype)
        if n_samp < args.min_samples:
            dropped.append((label, n_samp))
            continue
        out_fn = args.out / f"{label}.maf"
        grp.drop(columns="_TYPE").to_csv(out_fn, sep="\t", index=False)
        kept.append((label, n_samp, len(grp)))

    kept.sort(key=lambda x: -x[1])
    print(f"\nKept {len(kept)} cohorts (>= {args.min_samples} samples):")
    for label, n_samp, n_mut in kept:
        print(f"  {label:<45} {n_samp:>6} samples  {n_mut:>8} muts")
    print(f"\nDropped {len(dropped)} types below threshold "
          f"({sum(n for _, n in dropped)} samples total).")


if __name__ == "__main__":
    main()
