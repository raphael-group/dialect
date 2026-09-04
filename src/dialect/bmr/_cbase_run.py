"""Run the vendored CBaSE and turn its output into DIALECT's data contract.

This module owns everything CBaSE-specific: the MAF -> CBaSE-input conversion, the
two-script subprocess invocation (anchored to the vendored ``external/CBaSE`` so it
is CWD-independent), and the extraction of ``count_matrix.csv`` / ``bmr_pmfs.csv``
from CBaSE's raw output. It is wrapped by :class:`dialect.bmr.cbase.CBaSEProvider`.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Sequence
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd

from dialect.bmr.base import SampleAxisError
from dialect.data.io import check_file_exists

_N_SAMPLES_FLAG = "--n-samples"
_PMF_ONLY_FLAG = "--pmf-only"
_GENES_FILE_FLAG = "--genes-file"


def _validate_n_samples(n_samples: int | None) -> int | None:
    """Validate an optional explicit cohort size before launching CBaSE."""
    if n_samples is None:
        return None
    if isinstance(n_samples, bool) or not isinstance(n_samples, Integral):
        msg = "n_samples must be a positive integer or None"
        raise TypeError(msg)
    validated = int(n_samples)
    if validated <= 0:
        msg = "n_samples must be a positive integer or None"
        raise ValueError(msg)
    return validated


def _load_sample_ids(
    sample_ids: Sequence[str] | str | Path,
) -> tuple[str, ...]:
    """Load and validate one exact ordered sample axis.

    String and :class:`~pathlib.Path` inputs name a UTF-8 text file containing
    exactly one sample identifier per line. Sequence inputs are consumed in their
    existing order. Identifiers are never stripped or sorted: surrounding
    whitespace, empty identifiers, and duplicates fail closed.
    """
    if isinstance(sample_ids, (str, Path)):
        sample_axis_path = Path(sample_ids)
        check_file_exists(str(sample_axis_path))
        values = sample_axis_path.read_text(encoding="utf-8").splitlines()
    elif isinstance(sample_ids, Sequence):
        values = list(sample_ids)
    else:
        msg = "sample_ids must be an ordered sequence or a path"
        raise SampleAxisError(msg)

    if not values:
        msg = "sample_ids must contain at least one sample identifier"
        raise SampleAxisError(msg)

    validated: list[str] = []
    seen: set[str] = set()
    for position, sample_id in enumerate(values):
        if not isinstance(sample_id, str):
            msg = f"sample_ids[{position}] must be a string"
            raise SampleAxisError(msg)
        if not sample_id or sample_id != sample_id.strip():
            msg = (
                f"sample_ids[{position}] must be nonempty and have no surrounding "
                "whitespace"
            )
            raise SampleAxisError(msg)
        if sample_id in seen:
            msg = f"sample_ids contains duplicate identifier {sample_id!r}"
            raise SampleAxisError(msg)
        seen.add(sample_id)
        validated.append(sample_id)
    return tuple(validated)


def _resolve_sample_axis(
    *,
    n_samples: int | None,
    sample_ids: Sequence[str] | str | Path | None,
) -> tuple[tuple[str, ...] | None, int | None]:
    """Bind an explicit denominator to the exact ordered sample axis it counts."""
    validated_n_samples = _validate_n_samples(n_samples)
    if sample_ids is None:
        if validated_n_samples is not None:
            msg = (
                "sample_ids is required when n_samples is explicit so zero-event "
                "samples can be represented in the generated count matrices"
            )
            raise SampleAxisError(msg)
        return None, None

    sample_axis = _load_sample_ids(sample_ids)
    axis_n_samples = len(sample_axis)
    if (
        validated_n_samples is not None
        and validated_n_samples != axis_n_samples
    ):
        msg = (
            f"n_samples ({validated_n_samples}) does not match the exact sample "
            f"axis length ({axis_n_samples})"
        )
        raise SampleAxisError(msg)
    return sample_axis, axis_n_samples


def _run_cbase_step(label: str, cmd: list[str]) -> None:
    """Run one CBaSE subprocess step, raising RuntimeError with context on failure."""
    environment = os.environ.copy()
    cbase_directory = Path(cmd[1]).resolve().parent
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        part
        for part in (cbase_directory.as_posix(), existing_pythonpath)
        if part
    )
    try:
        subprocess.run(cmd, check=True, env=environment)
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


def generate_bmr_using_cbase(  # noqa: PLR0913
    maf: str,
    out: str,
    reference: str,
    threshold: str,
    *,
    n_samples: int | None = None,
    pmf_only: bool = False,
) -> None:
    """Invoke CBaSE's params + qvals scripts to produce its raw background output.

    Args:
        maf: TCGA-style input MAF.
        out: Output directory.
        reference: Reference genome build understood by CBaSE.
        threshold: PMF tail-truncation cutoff passed to CBaSE qvals.
        n_samples: Complete cohort size, including mutation-free samples. If omitted,
            preserve CBaSE's legacy behavior of inferring the size from samples with
            retained mutations.
        pmf_only: Compute PMFs only for genes observed in the retained mutation table
            and skip CBaSE's unrelated simulated gene-selection q-values.
    """
    validated_n_samples = _validate_n_samples(n_samples)
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
    if validated_n_samples is not None:
        cbase_params_cmd.extend([_N_SAMPLES_FLAG, str(validated_n_samples)])
    cbase_qvals_cmd = [
        sys.executable,
        str(cbase_qvals_script),
        str(out),
        str(cbase_output_dir),
        str(threshold),
    ]
    _run_cbase_step("CBaSE params", cbase_params_cmd)
    if pmf_only:
        (cbase_output_dir / "q_values.txt").unlink(missing_ok=True)
        retained = pd.read_csv(
            cbase_output_dir / "kept_mutations.csv",
            sep="\t",
            usecols=["gene"],
        )
        observed_genes = sorted(set(retained["gene"].dropna().astype(str)))
        if not observed_genes:
            msg = "CBaSE retained no genes for PMF generation"
            raise ValueError(msg)
        genes_path = cbase_output_dir / "observed_genes.txt"
        genes_path.write_text("\n".join(observed_genes) + "\n", encoding="utf-8")
        cbase_qvals_cmd.extend(
            [_PMF_ONLY_FLAG, _GENES_FILE_FLAG, str(genes_path)],
        )
    _run_cbase_step("CBaSE qvals", cbase_qvals_cmd)


def generate_counts_from_cbase_output(
    out: str,
    *,
    sample_ids: Sequence[str] | str | Path | None = None,
) -> None:
    """Pivot CBaSE mutations into counts on an optional exact sample axis.

    When ``sample_ids`` is provided, both generated matrices preserve that exact
    order. Otherwise they use CBaSE's lexicographically ordered all-effect retained
    sample axis. Both paths include all-zero rows for cohort members with no retained
    missense or nonsense event. Every sample retained anywhere in CBaSE's mutation
    table must belong to an explicit supplied axis.
    """
    cbase_kept_mutations_fn = Path(out) / "CBaSE_output" / "kept_mutations.csv"

    sample_axis = _load_sample_ids(sample_ids) if sample_ids is not None else None
    mut_df = pd.read_csv(
        cbase_kept_mutations_fn,
        sep="\t",
        dtype={"sample": "string"},
        keep_default_na=False,
    )
    if mut_df["sample"].isna().any():
        msg = "CBaSE retained mutation table contains an empty sample identifier"
        raise ValueError(msg)
    retained_sample_ids = set(mut_df["sample"])
    if sample_axis is None:
        sample_axis = tuple(sorted(retained_sample_ids))
        if not sample_axis:
            msg = "CBaSE retained mutation table contains no sample identifiers"
            raise ValueError(msg)
        if any(
            not sample_id or sample_id != sample_id.strip()
            for sample_id in sample_axis
        ):
            msg = "CBaSE retained mutation table contains a padded sample identifier"
            raise ValueError(msg)
    else:
        sample_axis_set = set(sample_axis)
        outside_axis = sorted(retained_sample_ids - sample_axis_set)
        if outside_axis:
            preview = ", ".join(repr(sample) for sample in outside_axis[:5])
            msg = (
                "CBaSE retained mutation samples outside the exact sample axis: "
                f"{preview}"
            )
            raise SampleAxisError(msg)

    mut_df = mut_df[mut_df["effect"].isin(["missense", "nonsense"])]
    gene_level_df = mut_df.pivot_table(
        index="gene",
        columns="sample",
        aggfunc="size",
        fill_value=0,
    ).T
    gene_level_df = gene_level_df.reindex(sample_axis, fill_value=0)
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
    mut_df = mut_df.reindex(sample_axis, fill_value=0)
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


def generate_bmr_and_counts(  # noqa: PLR0913
    maf: str,
    out: str,
    reference: str,
    threshold: str,
    *,
    n_samples: int | None = None,
    sample_ids: Sequence[str] | str | Path | None = None,
    pmf_only: bool = False,
) -> None:
    """Run CBaSE end-to-end on an optional exact, zero-complete sample axis."""
    sample_axis, resolved_n_samples = _resolve_sample_axis(
        n_samples=n_samples,
        sample_ids=sample_ids,
    )
    check_file_exists(maf)
    cbase_options: dict[str, int | bool | None] = {
        "n_samples": resolved_n_samples,
    }
    if pmf_only:
        cbase_options["pmf_only"] = True
    generate_bmr_using_cbase(
        maf,
        out,
        reference,
        threshold,
        **cbase_options,
    )
    generate_counts_from_cbase_output(out, sample_ids=sample_axis)
    generate_bmr_files_from_cbase_output(out)
