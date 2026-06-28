"""Split a cohort MAF into subtype-specific MAFs for stratified DIALECT runs.

Uses the official cBioPortal molecular SUBTYPE from data_clinical_patient.txt:
  UCEC: UCEC_POLE / UCEC_MSI / UCEC_CN_HIGH / UCEC_CN_LOW (the reviewer's four subtypes)
  BRCA: BRCA_LumA / BRCA_LumB / BRCA_Basal / BRCA_Her2 / BRCA_Normal (PAM50)

R1-1's argument: CBaSE's single global BMR is miscalibrated because a cohort pools
subtypes whose per-sample burdens differ by orders of magnitude; analyzing within a
(more burden-homogeneous) subtype should de-saturate the CO calls -- most in the
non-hypermutated subtypes (UCEC CN-low); within-POLE hypermutators still need the
sample-specific BMR.

Writes data/mafs_subtype/<COHORT>_<SUBTYPE>.maf. Run the pipeline per subtype via
  MAF_DIR=data/mafs_subtype ROOT=output/subtype \
    bash scripts/run_cohort_pipeline.sh UCEC_POLE

Usage:  python scripts/split_cohort_by_subtype.py UCEC
"""

from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path

import pandas as pd

OUT = Path("data/mafs_subtype")
DOWNLOADS = Path.home() / "Downloads"


def _patient_clinical(study: str) -> pd.DataFrame:
    tarball = DOWNLOADS / f"{study}_tcga_pan_can_atlas_2018.tar.gz"
    member = f"{study}_tcga_pan_can_atlas_2018/data_clinical_patient.txt"
    txt = subprocess.run(["tar", "-xzf", str(tarball), "-O", member],
                         capture_output=True, text=True, check=False).stdout
    return pd.read_csv(io.StringIO(txt), sep="\t", comment="#", low_memory=False)


def label_official(cohort: str, maf: pd.DataFrame) -> dict:
    """Sample -> official molecular SUBTYPE (from patient clinical), prefix stripped."""
    clin = _patient_clinical(cohort.lower())
    sub = dict(zip(clin["PATIENT_ID"], clin["SUBTYPE"].astype(str), strict=False))
    prefix = f"{cohort}_"
    out = {}
    for s in maf["Tumor_Sample_Barcode"].unique():
        patient = "-".join(s.split("-")[:3])  # TCGA-XX-XXXX-01 -> TCGA-XX-XXXX
        lab = sub.get(patient, "")
        out[s] = lab.replace(prefix, "") if lab and lab != "nan" else "NA"
    return out


def main() -> None:
    """Write per-subtype MAFs for the requested cohort."""
    cohort = (sys.argv[1] if len(sys.argv) > 1 else "UCEC").upper()
    maf = pd.read_csv(f"data/mafs_pancan/{cohort}.maf", sep="\t", low_memory=False)
    labels = label_official(cohort, maf)
    maf["__sub"] = maf["Tumor_Sample_Barcode"].map(labels)
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"{cohort} subtype split (official SUBTYPE):")
    for sub, grp in maf.groupby("__sub"):
        n = grp["Tumor_Sample_Barcode"].nunique()
        if sub in ("NA", "nan", "None") or n < 10:
            print(f"  skip {sub}: {n} samples")
            continue
        fn = OUT / f"{cohort}_{sub}.maf"
        grp.drop(columns="__sub").to_csv(fn, sep="\t", index=False)
        print(f"  {cohort}_{sub}: {n} samples, {len(grp)} muts -> {fn}")


if __name__ == "__main__":
    main()
