"""Run the vendored CBaSE and turn its output into DIALECT's data contract.

This module owns everything CBaSE-specific: the MAF -> CBaSE-input conversion, the
two-script subprocess invocation (anchored to the vendored ``external/CBaSE`` so it
is CWD-independent), and the extraction of ``count_matrix.csv`` / ``bmr_pmfs.csv``
from CBaSE's raw output. It is wrapped by :class:`dialect.bmr.cbase.CBaSEProvider`.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from dialect.data.io import check_file_exists


def _run_cbase_step(label: str, cmd: list[str]) -> None:
    """Run one CBaSE subprocess step, raising RuntimeError with context on failure."""
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as err:
        msg = (
            f"{label} step failed (exit {err.returncode}).\n"
            f"Command: {' '.join(cmd)}"
        )
        raise RuntimeError(msg) from err


def convert_maf_to_cbase_input_file(maf: str, dout: str) -> Path:
    """Project a TCGA-style MAF onto the 6-column TSV CBaSE expects."""
    maf_df = pd.read_csv(maf, sep="\t", low_memory=False)
    maf_df = maf_df.rename(
        columns={
            "Chromosome": "CHROM",
            "Start_Position": "POS",
            "Entrez_Gene_Id": "ID",
            "Reference_Allele": "REF",
            "Tumor_Seq_Allele2": "ALT",
            "Tumor_Sample_Barcode": "SAMPLE_BARCODE",
        },
    )[["CHROM", "POS", "ID", "REF", "ALT", "SAMPLE_BARCODE"]]
    fout = Path(dout) / "cbase_input.tsv"
    maf_df.to_csv(fout, sep="\t", header=False, index=False)
    return fout


def generate_bmr_using_cbase(
    maf: str,
    out: str,
    reference: str,
    threshold: str,
) -> None:
    """Invoke CBaSE's params + qvals scripts to produce its raw background output."""
    cbase_input_fn = convert_maf_to_cbase_input_file(maf, out)

    cbase_output_dir = Path(out) / "CBaSE_output"
    cbase_output_dir.mkdir(parents=True, exist_ok=True)

    # Anchor the vendored CBaSE to the repo root (parents[3] of this file) so the
    # invocation does not depend on the current working directory.
    cbase_dir = Path(__file__).resolve().parents[3] / "external" / "CBaSE"
    cbase_params_script = cbase_dir / "CBaSE_params_v1.2.py"
    cbase_qvals_script = cbase_dir / "CBaSE_qvals_v1.2.py"
    cbase_auxiliary_dir = cbase_dir / "auxiliary"

    # Use the interpreter running DIALECT (which has numpy/scipy) rather than a
    # bare "python" resolved from PATH.
    cbase_params_cmd = [
        sys.executable,
        str(cbase_params_script),
        str(cbase_input_fn),
        "1",
        str(reference),
        "3",
        "0",
        str(out),
        str(cbase_auxiliary_dir),
        str(cbase_output_dir),
    ]
    cbase_qvals_cmd = [
        sys.executable,
        str(cbase_qvals_script),
        str(out),
        str(cbase_output_dir),
        str(threshold),
    ]
    for label, cmd in (
        ("CBaSE params", cbase_params_cmd),
        ("CBaSE qvals", cbase_qvals_cmd),
    ):
        _run_cbase_step(label, cmd)


def generate_counts_from_cbase_output(out: str) -> None:
    """Pivot CBaSE's kept-mutations table into gene-level and effect-level counts."""
    cbase_kept_mutations_fn = Path(out) / "CBaSE_output" / "kept_mutations.csv"

    mut_df = pd.read_csv(cbase_kept_mutations_fn, sep="\t")
    mut_df = mut_df[mut_df["effect"].isin(["missense", "nonsense"])]
    gene_level_df = mut_df.pivot_table(
        index="gene",
        columns="sample",
        aggfunc="size",
        fill_value=0,
    ).T
    gene_level_df.to_csv(
        Path(out) / "gene_level_count_matrix.csv",
        index=True,
    )
    mut_df["gene"] = mut_df["gene"] + "_" + mut_df["effect"].str[0].str.upper()
    mut_df = mut_df.pivot_table(
        index="gene",
        columns="sample",
        aggfunc="size",
        fill_value=0,
    ).T
    mut_df.to_csv(
        Path(out) / "count_matrix.csv",
        index=True,
    )


def generate_bmr_files_from_cbase_output(out: str) -> None:
    """Reshape CBaSE's per-gene missense/nonsense PMFs into ``bmr_pmfs.csv``."""
    mis_bmr_fn = Path(out) / "CBaSE_output" / "pofmigivens.txt"
    non_bmr_fn = Path(out) / "CBaSE_output" / "pofkigivens.txt"

    all_dfs = []
    for fn, suffix in zip(
        [mis_bmr_fn, non_bmr_fn], ["_M", "_N"], strict=True,
    ):
        with fn.open() as f:
            lines = f.readlines()
            file_length = len(lines)
            max_cols = np.max([len(line.split("\t")) for line in lines])

        column_names = ["gene", *list(range(max_cols))]
        pmf_df = pd.read_csv(
            fn,
            sep="\t",
            names=column_names,
            skiprows=range(0, file_length, 2),
        )

        # Modify gene names with the appropriate suffix
        pmf_df["gene"] = pmf_df["gene"].str.rsplit("_", n=1).str[0] + suffix

        pmf_df = pmf_df.set_index("gene", drop=True)
        all_dfs.append(pmf_df)

    pmf_df = pd.concat(all_dfs)
    pmf_df.to_csv(Path(out) / "bmr_pmfs.csv", index=True)


def generate_bmr_and_counts(
    maf: str,
    out: str,
    reference: str,
    threshold: str,
) -> None:
    """Run CBaSE end-to-end: MAF -> background PMFs + count matrices on disk."""
    check_file_exists(maf)
    generate_bmr_using_cbase(maf, out, reference, threshold)
    generate_bmr_files_from_cbase_output(out)
    generate_counts_from_cbase_output(out)
