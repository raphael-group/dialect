"""Group the 69-cohort consensus networks into cancer-type families for lit search.

Reads /tmp/consensus_networks.json (per-cohort ME/CO edges), maps each cohort to a
canonical cancer-type group, pools edges across the group's constituent cohorts
(preserving per-cohort/BMR/direction provenance and cross-cohort recurrence), tags
each gene as an OncoKB cancer gene, and classifies each pair as driver-driver (dd),
driver-passenger (dp), or passenger-passenger (pp). Emits one record per cancer-type
group, ready to hand to a literature-validation agent.

Output: /tmp/cancer_type_groups.json
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

# cohort key (study::cohort or just cohort for TCGA) -> cancer-type group
GROUP_MAP = {
    # Breast
    "TCGA::BRCA": "Breast", "MSK-IMPACT2026::Breast_Cancer": "Breast",
    "MSK-CHORD2024::Breast_Cancer": "Breast",
    # Lung NSCLC
    "TCGA::LUAD": "Lung_NSCLC", "TCGA::LUSC": "Lung_NSCLC",
    "MSK-IMPACT2026::Non_Small_Cell_Lung_Cancer": "Lung_NSCLC",
    "MSK-CHORD2024::Non_Small_Cell_Lung_Cancer": "Lung_NSCLC",
    # Lung SCLC
    "MSK-IMPACT2026::Small_Cell_Lung_Cancer": "Lung_SCLC",
    # Colorectal
    "TCGA::CRAD": "Colorectal", "MSK-IMPACT2026::Colorectal_Cancer": "Colorectal",
    "MSK-CHORD2024::Colorectal_Cancer": "Colorectal",
    # Pancreatic
    "TCGA::PAAD": "Pancreatic", "MSK-IMPACT2026::Pancreatic_Cancer": "Pancreatic",
    "MSK-CHORD2024::Pancreatic_Cancer": "Pancreatic",
    # Prostate
    "TCGA::PRAD": "Prostate", "MSK-IMPACT2026::Prostate_Cancer": "Prostate",
    "MSK-CHORD2024::Prostate_Cancer": "Prostate",
    # Glioma group: lower-grade, GBM, and MSK glioma/CNS cohorts
    "TCGA::LGG": "Glioma", "TCGA::GBM": "Glioma",
    "MSK-IMPACT2026::Glioma": "Glioma", "MSK-IMPACT2026::CNS_Cancer": "Glioma",
    # Melanoma
    "TCGA::SKCM": "Melanoma", "MSK-IMPACT2026::Melanoma": "Melanoma",
    "TCGA::UVM": "Uveal_Melanoma",
    "MSK-IMPACT2026::Skin_Cancer_Non_Melanoma": "Skin_NonMelanoma",
    # Uterine / Endometrial
    "TCGA::UCS": "Uterine", "MSK-IMPACT2026::Endometrial_Cancer": "Uterine",
    "MSK-IMPACT2026::Uterine_Sarcoma": "Uterine",
    # Ovarian
    "TCGA::OV": "Ovarian", "MSK-IMPACT2026::Ovarian_Cancer": "Ovarian",
    # Gastroesophageal
    "TCGA::STAD": "Gastroesophageal", "TCGA::ESCA": "Gastroesophageal",
    "MSK-IMPACT2026::Esophagogastric_Cancer": "Gastroesophageal",
    # Head & Neck
    "TCGA::HNSC": "Head_Neck", "MSK-IMPACT2026::Head_and_Neck_Cancer": "Head_Neck",
    "MSK-IMPACT2026::Salivary_Gland_Cancer": "Salivary",
    # Bladder / urothelial
    "TCGA::BLCA": "Bladder", "MSK-IMPACT2026::Bladder_Cancer": "Bladder",
    # Renal
    "TCGA::KIRC": "Renal", "TCGA::KIRP": "Renal", "TCGA::KICH": "Renal",
    "MSK-IMPACT2026::Renal_Cell_Carcinoma": "Renal",
    # Liver / biliary
    "TCGA::LIHC": "Liver_Biliary", "TCGA::CHOL": "Liver_Biliary",
    "MSK-IMPACT2026::Hepatobiliary_Cancer": "Liver_Biliary",
    # Thyroid
    "TCGA::THCA": "Thyroid", "MSK-IMPACT2026::Thyroid_Cancer": "Thyroid",
    # AML
    "TCGA::LAML": "AML",
    # Sarcoma (soft tissue + bone + GIST + nerve sheath + PNS)
    "TCGA::SARC": "Sarcoma", "MSK-IMPACT2026::Soft_Tissue_Sarcoma": "Sarcoma",
    "MSK-IMPACT2026::Bone_Cancer": "Sarcoma",
    "MSK-IMPACT2026::Gastrointestinal_Stromal_Tumor": "GIST",
    "MSK-IMPACT2026::Nerve_Sheath_Tumor": "Sarcoma",
    "MSK-IMPACT2026::Peripheral_Nervous_System": "Sarcoma",
    # Cervical
    "TCGA::CESC": "Cervical", "MSK-IMPACT2026::Cervical_Cancer": "Cervical",
    # Mesothelioma
    "TCGA::MESO": "Mesothelioma", "MSK-IMPACT2026::Mesothelioma": "Mesothelioma",
    # Germ cell
    "TCGA::TGCT": "Germ_Cell", "MSK-IMPACT2026::Germ_Cell_Tumor": "Germ_Cell",
    # Endocrine / rare
    "TCGA::ACC": "Adrenocortical", "TCGA::PCPG": "Pheo_Paraganglioma",
    "TCGA::DLBC": "Lymphoma_DLBCL",
    "MSK-IMPACT2026::Gastrointestinal_Neuroendocrine_Tumor": "GI_NeuroendocrineV",
    # MSK-only GI / other
    "MSK-IMPACT2026::Ampullary_Cancer": "Ampullary",
    "MSK-IMPACT2026::Anal_Cancer": "Anal",
    "MSK-IMPACT2026::Appendiceal_Cancer": "Appendiceal",
    "MSK-IMPACT2026::Small_Bowel_Cancer": "Small_Bowel",
    "MSK-IMPACT2026::Cancer_of_Unknown_Primary": "CUP",
}


def load_drivers() -> set[str]:
    """Return the set of OncoKB cancer-gene Hugo symbols."""
    df = pd.read_csv("data/references/OncoKB_Cancer_Gene_List.tsv", sep="\t")
    return set(df["Hugo Symbol"].astype(str))


def classify(a: str, b: str, drivers: set[str]) -> str:
    """Classify a gene pair as dd, dp, or pp by OncoKB driver membership."""
    da, db = a in drivers, b in drivers
    return "dd" if da and db else "pp" if not (da or db) else "dp"


def main() -> None:
    """Group consensus networks by cancer type and write the grouped JSON."""
    records = json.loads(Path("/tmp/consensus_networks.json").read_text())
    drivers = load_drivers()

    # group -> direction -> pair -> aggregate
    groups: dict = defaultdict(lambda: {
        "cohorts": [], "ME": defaultdict(list), "CO": defaultdict(list),
    })
    for r in records:
        ckey = f"{r['study']}::{r['cohort']}"
        g = GROUP_MAP.get(ckey)
        if g is None:
            print(f"WARN unmapped cohort: {ckey}")
            continue
        groups[g]["cohorts"].append({
            "study": r["study"], "cohort": r["cohort"],
            "n_samples": r["n_samples"], "median_tmb": r["median_tmb"],
        })
        for direction in ("ME", "CO"):
            for p in r[direction]:
                groups[g][direction][p["pair"]].append({
                    "study": r["study"], "cohort": r["cohort"],
                    "supported_by": p["supported_by"], "best_rank": p["best_rank"],
                })

    out = []
    for g, data in sorted(groups.items()):
        rec = {"group": g, "cohorts": data["cohorts"], "ME": [], "CO": []}
        for direction in ("ME", "CO"):
            pairs = []
            for pair, prov in data[direction].items():
                a, b = pair.split(":")
                cls = classify(a, b, drivers)
                # union of all supporting BMRs across provenance
                bmrs = sorted({x for pr in prov for x in pr["supported_by"]})
                pairs.append({
                    "pair": pair, "class": cls,
                    "n_cohorts": len({pr["cohort"] + pr["study"] for pr in prov}),
                    "supported_by": bmrs,
                    "best_rank": min(pr["best_rank"] for pr in prov),
                    "provenance": prov,
                })
            # dd first, then dp, then by cross-cohort recurrence
            order = {"dd": 0, "dp": 1, "pp": 2}
            pairs.sort(
                key=lambda d: (order[d["class"]], -d["n_cohorts"], d["best_rank"]),
            )
            rec[direction] = pairs
        out.append(rec)

    Path("/tmp/cancer_type_groups.json").write_text(json.dumps(out, indent=2))

    # summary
    print(f"\ncancer-type groups: {len(out)}")
    print(f"{'group':22} {'#cohorts':>8} {'ME(dd/dp/pp)':>16} {'CO(dd/dp/pp)':>16}")
    tot = {"dd": 0, "dp": 0, "pp": 0}
    def counts(rec: dict, direction: str) -> dict[str, int]:
        c = {"dd": 0, "dp": 0, "pp": 0}
        for p in rec[direction]:
            c[p["class"]] += 1
            tot[p["class"]] += 1
        return c
    for rec in out:
        me, co = counts(rec, "ME"), counts(rec, "CO")
        print(f"{rec['group']:22} {len(rec['cohorts']):8d} "
              f"{me['dd']:4d}/{me['dp']:3d}/{me['pp']:3d}    "
              f"{co['dd']:4d}/{co['dp']:3d}/{co['pp']:3d}")
    print(f"\nTOTAL pair-instances by class: {tot}")


if __name__ == "__main__":
    main()
