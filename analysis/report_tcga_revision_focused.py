"""Build the focused DIALECT revision tables, runtime summary, and Figure 6.

This stage consumes only the validated matched K=500 fit, provider-specific
postprocessing, prespecified calibration, and frozen global reporting rule.  It
does not choose thresholds or modify scientific results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import calibrate_tcga_revision_focused as calibration
from analysis import freeze_tcga_revision_reporting_rule as rule_module
from analysis import postprocess_tcga_revision_focused as postprocess
from analysis import run_tcga_revision_k500 as core
from analysis.prepare_tcga_revision_focused import validate_provider_root
from dialect.data.tcga import TCGA_COHORTS

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

SCHEMA_VERSION: Final = "1.0.0"
REPORT_CONTRACT: Final = "focused-revision-reporting-artifacts-v1"
HIGH_BURDEN_QUANTILE: Final = 0.99
EXPECTED_TUMOR_COUNT: Final = 10_433
FOCAL_BURDEN_COHORT: Final = "UCEC"
PROVIDER_LABELS: Final = {
    "cbase": "CBaSE",
    "dig": "DIG",
    "mutsig": "MutSig",
}
PROVIDER_COLORS: Final = {
    "cbase": "#6B7280",
    "dig": "#0072B2",
    "mutsig": "#D55E00",
}


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path, *, relative_to: Path) -> dict[str, int | str]:
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _write_atomic(path: Path, content: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_rule(
    rule_path: Path,
    calibration_root: Path,
    postprocess_root: Path,
) -> dict[str, Any]:
    rule = json.loads(rule_path.read_text(encoding="utf-8"))
    if (
        rule.get("schema_version") != SCHEMA_VERSION
        or rule.get("contract") != rule_module.RULE_CONTRACT
        or rule.get("scope") != "one-identical-rule-across-all-32-cancer-types"
        or rule.get("primary_provider") != "mutsig"
        or rule.get("continuity_provider") != "cbase"
        or rule.get("supplementary_providers") != ["dig"]
        or rule.get("threshold_comparison") != "inclusive-less-than-or-equal"
        or rule.get("thresholds_selected_from_observed_pairs") is not False
        or rule.get("calibration_summary_sha256")
        != _sha256(calibration_root / calibration.SUMMARY_NAME)
        or rule.get("postprocess_manifest_sha256")
        != _sha256(postprocess_root / postprocess.ROOT_MANIFEST_NAME)
    ):
        msg = "Frozen reporting rule is invalid or bound to different inputs."
        raise ValueError(msg)
    for key in ("primary_q_threshold", "sensitivity_q_threshold"):
        value = rule.get(key)
        if not isinstance(value, (int, float)) or not 0 < float(value) < 1:
            msg = f"Frozen reporting rule has invalid {key}."
            raise ValueError(msg)
    return rule


def _read_inference(postprocess_root: Path, cohort: str) -> pd.DataFrame:
    frame = pd.read_csv(
        postprocess_root / cohort / postprocess.RESULT_NAME,
        float_precision="round_trip",
    )
    if frame[["gene_a", "gene_b"]].duplicated().any():
        msg = f"Duplicate pair in postprocessed results: {cohort}"
        raise ValueError(msg)
    return frame


def _read_counts(provider_root: Path, cohort: str) -> pd.DataFrame:
    frame = pd.read_csv(
        provider_root / "cohorts" / cohort / "count_matrix.csv",
        index_col=0,
    )
    values = frame.to_numpy(dtype=np.float64)
    if (
        frame.empty
        or not frame.index.is_unique
        or not frame.columns.is_unique
        or not np.isfinite(values).all()
        or (values < 0).any()
        or not np.equal(values, np.floor(values)).all()
    ):
        msg = f"Invalid focused count matrix: {cohort}"
        raise ValueError(msg)
    return frame


def _burden_values(
    provider_root: Path,
    cohorts: Sequence[str],
) -> dict[str, np.ndarray]:
    return {
        cohort: _read_counts(provider_root, cohort).sum(axis=1).to_numpy(dtype=float)
        for cohort in cohorts
    }


def _high_burden_threshold(values: Mapping[str, np.ndarray]) -> float:
    pooled = np.concatenate(tuple(values.values()))
    if len(pooled) != EXPECTED_TUMOR_COUNT or not np.isfinite(pooled).all():
        msg = "Cannot define the pooled high-burden threshold."
        raise ValueError(msg)
    return float(np.quantile(pooled, HIGH_BURDEN_QUANTILE, method="higher"))


def _cohort_burden_source(values: Mapping[str, np.ndarray]) -> pd.DataFrame:
    """Return deidentified values underlying every burden summary in Table S5."""
    frames = []
    for cohort in TCGA_COHORTS:
        burdens = np.asarray(values[cohort], dtype=float)
        frames.append(
            pd.DataFrame(
                {
                    "cohort": cohort,
                    "cohort_row": np.arange(1, len(burdens) + 1, dtype=np.int64),
                    "pre_k_total_nonsynonymous_snv_event_count": burdens,
                },
            ),
        )
    return pd.concat(frames, ignore_index=True)


def _cohort_summary_row(  # noqa: PLR0913
    *,
    cohort: str,
    frame: pd.DataFrame,
    burdens: np.ndarray,
    high_burden_threshold: float,
    primary_q: float,
    sensitivity_q: float,
) -> dict[str, int | float | str]:
    row: dict[str, int | float | str] = {
        "cohort": cohort,
        "tumors": len(burdens),
        "selected_events": 500,
        "tested_pairs": len(frame),
        "burden_median": float(np.median(burdens)),
        "burden_q25": float(np.quantile(burdens, 0.25)),
        "burden_q75": float(np.quantile(burdens, 0.75)),
        "burden_p90": float(np.quantile(burdens, 0.90)),
        "burden_p95": float(np.quantile(burdens, 0.95)),
        "burden_max": float(np.max(burdens)),
        "high_burden_fraction": float((burdens >= high_burden_threshold).mean()),
    }
    for provider in core.BMRS:
        q_values = frame[f"{provider}_q_value"].to_numpy(dtype=float)
        directions = frame[f"{provider}_direction"].astype("string")
        for label, threshold in (
            ("primary", primary_q),
            ("sensitivity", sensitivity_q),
        ):
            significant = q_values <= threshold
            row[f"{provider}_{label}_total"] = int(significant.sum())
            row[f"{provider}_{label}_me"] = int(
                (significant & directions.eq("ME").to_numpy()).sum(),
            )
            row[f"{provider}_{label}_co"] = int(
                (significant & directions.eq("CO").to_numpy()).sum(),
            )
    return row


def _top_primary_pairs(
    frame: pd.DataFrame,
    *,
    cohort: str,
    primary_q: float,
    per_direction: int = 10,
) -> pd.DataFrame:
    provider = "mutsig"
    significant = frame[f"{provider}_q_value"] <= primary_q
    parts = []
    for direction in ("ME", "CO"):
        selected = frame.loc[
            significant & frame[f"{provider}_direction"].eq(direction),
        ].copy()
        selected["absolute_mutsig_rho"] = selected["mutsig_rho"].abs()
        selected = selected.sort_values(
            ["mutsig_q_value", "mutsig_p_value", "absolute_mutsig_rho"],
            ascending=[True, True, False],
            kind="stable",
        ).head(per_direction)
        selected.insert(0, "direction", direction)
        parts.append(selected)
    result = pd.concat(parts, ignore_index=True)
    result.insert(0, "cohort", cohort)
    for provider in core.BMRS:
        result[f"{provider}_primary_significant"] = (
            result[f"{provider}_q_value"] <= primary_q
        )
    result["provider_support"] = sum(
        result[f"{provider}_primary_significant"].astype(np.int8)
        for provider in core.BMRS
    )
    return result.drop(columns="absolute_mutsig_rho")


def _overlap_rows(
    frame: pd.DataFrame,
    *,
    cohort: str,
    primary_q: float,
) -> list[dict[str, int | float | str]]:
    rows = []
    for direction in ("ME", "CO"):
        masks = {
            provider: (
                (frame[f"{provider}_q_value"].to_numpy(dtype=float) <= primary_q)
                & frame[f"{provider}_direction"].eq(direction).to_numpy()
            )
            for provider in core.BMRS
        }
        support = sum(mask.astype(np.int8) for mask in masks.values())
        rows.append(
            {
                "cohort": cohort,
                "direction": direction,
                "q_threshold": primary_q,
                "cbase": int(masks["cbase"].sum()),
                "dig": int(masks["dig"].sum()),
                "mutsig": int(masks["mutsig"].sum()),
                "at_least_one": int((support >= 1).sum()),
                "at_least_two": int((support >= 2).sum()),
                "all_three": int((support == 3).sum()),
                "mutsig_and_cbase": int((masks["mutsig"] & masks["cbase"]).sum()),
            },
        )
    return rows


def _runtime_rows(run_root: Path, cohorts: Sequence[str]) -> list[dict[str, Any]]:
    rows = []
    for cohort in cohorts:
        for provider in core.BMRS:
            path = run_root / "tasks" / cohort / provider / "task_manifest.json"
            manifest = json.loads(path.read_text(encoding="utf-8"))
            usage = manifest["resource_usage"]
            rows.append(
                {
                    "cohort": cohort,
                    "provider": provider,
                    "pairwise_rows": manifest["pairwise_rows"],
                    "elapsed_seconds": usage["elapsed_seconds"],
                    "user_cpu_seconds": usage["user_cpu_seconds"],
                    "system_cpu_seconds": usage["system_cpu_seconds"],
                    "peak_rss_bytes": usage["peak_rss"]["bytes"],
                },
            )
    return rows


def _pmf_mean(pmf: Mapping[int, float]) -> float:
    return float(sum(int(key) * float(value) for key, value in pmf.items()))


def _expected_selected_burden(
    *,
    run_root: Path,
    cohort: str,
    provider: str,
) -> tuple[np.ndarray, np.ndarray]:
    contract = json.loads(
        (run_root / "contracts" / f"{cohort}.json").read_text(encoding="utf-8"),
    )
    counts, pmfs = core._load_frozen_scientific_inputs(contract, provider)  # noqa: SLF001
    features = tuple(contract["features"])
    selected = counts.loc[:, features]
    single = pd.read_csv(
        run_root / "tasks" / cohort / provider / "single_gene_results.csv",
        float_precision="round_trip",
    )
    if single["Gene Name"].tolist() != list(features):
        msg = f"Single-event axis changed: {cohort}/{provider}"
        raise ValueError(msg)
    pi = single["Pi"].to_numpy(dtype=float)
    expected = np.zeros(len(selected), dtype=float)
    for feature, fitted_pi in zip(features, pi, strict=True):
        background = pmfs[feature]
        if isinstance(background, dict):
            expected += _pmf_mean(background) + fitted_pi
        else:
            if len(background) != len(selected):
                msg = f"Sample-specific PMF axis changed: {cohort}/{provider}"
                raise ValueError(msg)
            expected += np.fromiter(
                (_pmf_mean(item) for item in background),
                dtype=float,
                count=len(selected),
            ) + fitted_pi
    return selected.sum(axis=1).to_numpy(dtype=float), expected


def _figure6_burden_source(run_root: Path) -> pd.DataFrame:
    """Return the deidentified numeric values plotted in Figure 6 panel A."""
    frame: pd.DataFrame | None = None
    for provider in core.BMRS:
        observed, expected = _expected_selected_burden(
            run_root=run_root,
            cohort=FOCAL_BURDEN_COHORT,
            provider=provider,
        )
        if frame is None:
            frame = pd.DataFrame(
                {
                    "cohort": FOCAL_BURDEN_COHORT,
                    "cohort_row": np.arange(1, len(observed) + 1, dtype=np.int64),
                    "observed_selected_event_count": observed,
                },
            )
        elif not np.array_equal(
            frame["observed_selected_event_count"].to_numpy(dtype=float),
            observed,
        ):
            msg = "Observed selected burden differs between providers."
            raise ValueError(msg)
        frame[f"{provider}_model_expected_selected_event_count"] = expected
    if frame is None:
        msg = "Figure 6 burden source could not be constructed."
        raise RuntimeError(msg)
    return frame


def _plot_figure6(
    *,
    burden_source: pd.DataFrame,
    summary: pd.DataFrame,
    calibration_table: pd.DataFrame,
    primary_q: float,
    output: Path,
) -> None:
    mpl.rcParams.update(
        {
            "axes.spines.right": False,
            "axes.spines.top": False,
            "font.size": 9,
            "figure.dpi": 150,
        },
    )
    figure, axes = plt.subplots(2, 2, figsize=(13, 12), constrained_layout=True)

    ax = axes[0, 0]
    observed = burden_source["observed_selected_event_count"].to_numpy(dtype=float)
    for provider in core.BMRS:
        expected = burden_source[
            f"{provider}_model_expected_selected_event_count"
        ].to_numpy(dtype=float)
        ax.scatter(
            observed + 1,
            expected + 1,
            s=10,
            alpha=0.45,
            color=PROVIDER_COLORS[provider],
            label=PROVIDER_LABELS[provider],
        )
    maximum = float(max(ax.get_xlim()[1], ax.get_ylim()[1]))
    ax.plot([1, maximum], [1, maximum], color="#111827", linewidth=0.8, linestyle="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Observed selected-event count per tumor + 1")
    ax.set_ylabel("Model-expected count per tumor + 1")
    ax.set_title("A  UCEC burden across background models", loc="left")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    ordered = summary.sort_values("mutsig_primary_co", ascending=True)
    positions = np.arange(len(ordered))
    offsets = {"cbase": -0.22, "dig": 0.0, "mutsig": 0.22}
    provider_counts = []
    for provider in core.BMRS:
        counts = ordered[f"{provider}_primary_co"].to_numpy(dtype=float)
        provider_counts.append(counts)
        ax.scatter(
            np.log10(counts + 1),
            positions + offsets[provider],
            s=24,
            color=PROVIDER_COLORS[provider],
            label=PROVIDER_LABELS[provider],
        )
    ax.set_yticks(positions, ordered["cohort"])
    maximum_count = max(float(values.max()) for values in provider_counts)
    candidate_ticks = np.asarray([0, 1, 10, 100, 1_000, 10_000, 100_000])
    count_ticks = candidate_ticks[candidate_ticks <= maximum_count]
    if len(count_ticks) == 0 or count_ticks[-1] < maximum_count:
        count_ticks = np.append(count_ticks, int(np.ceil(maximum_count)))
    ax.set_xticks(
        np.log10(count_ticks + 1),
        [f"{value:,}" for value in count_ticks],
    )
    ax.set_xlabel(f"CO pairs at q <= {primary_q:g}")
    ax.set_title("B  Co-occurrence calls across background models", loc="left")
    ax.grid(axis="x", alpha=0.2)

    ax = axes[1, 0]
    marginal = calibration_table.loc[
        calibration_table["screen"].eq("marginal_lrt"),
    ]
    for provider in core.BMRS:
        selected = marginal.loc[marginal["provider"].eq(provider)]
        for threshold, group in selected.groupby("threshold"):
            x = np.full(len(group), float(threshold))
            ax.scatter(
                x,
                group["rate"],
                s=22,
                alpha=0.75,
                color=PROVIDER_COLORS[provider],
            )
        means = selected.groupby("threshold", as_index=False)["rate"].mean()
        ax.plot(
            means["threshold"],
            means["rate"],
            color=PROVIDER_COLORS[provider],
            label=PROVIDER_LABELS[provider],
        )
    limits = [0, max(0.11, float(marginal["rate"].max()) * 1.05)]
    ax.plot(limits, limits, color="#111827", linewidth=0.8, linestyle="--")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel("Nominal p-value threshold")
    ax.set_ylabel("Null rejection rate")
    ax.set_title("C  Profile-LRT null calibration", loc="left")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    family = calibration_table.loc[
        calibration_table["screen"].eq("complete_null_bh_family"),
    ]
    for provider in core.BMRS:
        selected = family.loc[family["provider"].eq(provider)]
        for threshold, group in selected.groupby("threshold"):
            ax.scatter(
                np.full(len(group), float(threshold)),
                group["rate"],
                s=22,
                alpha=0.75,
                color=PROVIDER_COLORS[provider],
            )
        means = selected.groupby("threshold", as_index=False)["rate"].mean()
        ax.plot(
            means["threshold"],
            means["rate"],
            color=PROVIDER_COLORS[provider],
            label=PROVIDER_LABELS[provider],
        )
    limits = [0, max(0.22, float(family["rate"].max()) * 1.05)]
    ax.plot(limits, limits, color="#111827", linewidth=0.8, linestyle="--")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel("Nominal BH q-value threshold")
    ax.set_ylabel("Complete-null families with >=1 rejection")
    ax.set_title("D  Complete-null family calibration", loc="left")
    ax.legend(frameon=False)

    figure.savefig(
        output,
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(figure)
    if not output.is_file() or output.stat().st_size == 0:
        msg = "Figure 6 rendering failed."
        raise RuntimeError(msg)


def validate_report(output_root: Path) -> dict[str, Any]:
    """Validate the immutable reporting tree and its core table dimensions."""
    manifest_path = output_root / "report_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_outputs = {
        "cohort_burden_source.csv",
        "figure6_burden_source.csv",
        "table_s5.csv",
        "provider_overlap.csv",
        "top_primary_pairs.csv",
        "runtime_summary.csv",
        "table_s5.tex",
        "figure6.pdf",
    }
    records = manifest.get("outputs", {})
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("contract") != REPORT_CONTRACT
        or manifest.get("cohorts") != list(TCGA_COHORTS)
        or manifest.get("primary_provider") != "mutsig"
        or manifest.get("high_burden_definition", {}).get("pooled_tumor_count")
        != EXPECTED_TUMOR_COUNT
        or set(records) != expected_outputs
        or {path.name for path in output_root.iterdir()}
        != {*expected_outputs, "report_manifest.json"}
    ):
        msg = "Focused reporting manifest or inventory is invalid."
        raise ValueError(msg)
    for name in expected_outputs:
        path = output_root / name
        record = records[name]
        if (
            record.get("path") != name
            or record.get("bytes") != path.stat().st_size
            or record.get("sha256") != _sha256(path)
        ):
            msg = f"Focused reporting output changed: {name}"
            raise ValueError(msg)
    summary = pd.read_csv(output_root / "table_s5.csv")
    overlap = pd.read_csv(output_root / "provider_overlap.csv")
    runtime = pd.read_csv(output_root / "runtime_summary.csv")
    cohort_burden = pd.read_csv(output_root / "cohort_burden_source.csv")
    figure_burden = pd.read_csv(output_root / "figure6_burden_source.csv")
    expected_cohort_burden_columns = {
        "cohort",
        "cohort_row",
        "pre_k_total_nonsynonymous_snv_event_count",
    }
    expected_figure_burden_columns = {
        "cohort",
        "cohort_row",
        "observed_selected_event_count",
        *(f"{provider}_model_expected_selected_event_count" for provider in core.BMRS),
    }
    expected_rows = cohort_burden.groupby("cohort", sort=False).cumcount() + 1
    cohort_burden_values = cohort_burden[
        "pre_k_total_nonsynonymous_snv_event_count"
    ].to_numpy(dtype=float)
    figure_burden_values = figure_burden.drop(
        columns=["cohort", "cohort_row"],
    ).to_numpy(dtype=float)
    if (
        len(summary) != len(TCGA_COHORTS)
        or summary["cohort"].tolist() != list(TCGA_COHORTS)
        or int(summary["tumors"].sum()) != EXPECTED_TUMOR_COUNT
        or len(overlap) != len(TCGA_COHORTS) * 2
        or len(runtime) != len(TCGA_COHORTS) * len(core.BMRS)
        or len(cohort_burden) != EXPECTED_TUMOR_COUNT
        or cohort_burden["cohort"].nunique() != len(TCGA_COHORTS)
        or set(cohort_burden.columns) != expected_cohort_burden_columns
        or set(figure_burden.columns) != expected_figure_burden_columns
        or not cohort_burden["cohort_row"].eq(expected_rows).all()
        or not summary.set_index("cohort")["tumors"].eq(
            cohort_burden["cohort"].value_counts(sort=False),
        ).all()
        or set(figure_burden["cohort"]) != {FOCAL_BURDEN_COHORT}
        or len(figure_burden)
        != int(summary.set_index("cohort").loc[FOCAL_BURDEN_COHORT, "tumors"])
        or not figure_burden["cohort_row"].eq(
            np.arange(1, len(figure_burden) + 1),
        ).all()
        or not np.isfinite(cohort_burden_values).all()
        or (cohort_burden_values < 0).any()
        or not np.equal(cohort_burden_values, np.floor(cohort_burden_values)).all()
        or not np.isfinite(figure_burden_values).all()
        or (figure_burden_values < 0).any()
        or "sample" in " ".join(cohort_burden.columns).casefold()
        or "sample" in " ".join(figure_burden.columns).casefold()
        or (output_root / "figure6.pdf").read_bytes()[:5] != b"%PDF-"
    ):
        msg = "Focused reporting tables or PDF failed dimensional validation."
        raise ValueError(msg)
    return manifest


def build_report(  # noqa: PLR0913
    *,
    run_root: Path,
    provider_root: Path,
    postprocess_root: Path,
    calibration_root: Path,
    rule_path: Path,
    output_root: Path,
) -> Path:
    """Build all result-dependent reporting artifacts once."""
    cohorts = tuple(TCGA_COHORTS)
    validate_provider_root(provider_root, cohorts)
    postprocess.validate_derived_root(postprocess_root, cohorts)
    calibration.validate_summary(calibration_root)
    rule = _load_rule(rule_path, calibration_root, postprocess_root)
    postprocess._validate_completion(run_root, cohorts)  # noqa: SLF001
    if output_root.exists() or output_root.is_symlink():
        msg = f"Refusing to overwrite reporting root: {output_root}"
        raise FileExistsError(msg)

    burdens = _burden_values(provider_root, cohorts)
    burden_threshold = _high_burden_threshold(burdens)
    primary_q = float(rule["primary_q_threshold"])
    sensitivity_q = float(rule["sensitivity_q_threshold"])
    summary_rows = []
    overlap_rows = []
    top_frames = []
    for cohort in cohorts:
        frame = _read_inference(postprocess_root, cohort)
        summary_rows.append(
            _cohort_summary_row(
                cohort=cohort,
                frame=frame,
                burdens=burdens[cohort],
                high_burden_threshold=burden_threshold,
                primary_q=primary_q,
                sensitivity_q=sensitivity_q,
            ),
        )
        overlap_rows.extend(
            _overlap_rows(frame, cohort=cohort, primary_q=primary_q),
        )
        top_frames.append(
            _top_primary_pairs(frame, cohort=cohort, primary_q=primary_q),
        )
    summary = pd.DataFrame(summary_rows)
    overlap = pd.DataFrame(overlap_rows)
    top_pairs = pd.concat(top_frames, ignore_index=True)
    runtime = pd.DataFrame(_runtime_rows(run_root, cohorts))
    cohort_burden_source = _cohort_burden_source(burdens)
    figure6_burden_source = _figure6_burden_source(run_root)
    calibration_table = pd.read_csv(
        calibration_root / calibration.SUMMARY_TABLE_NAME,
        float_precision="round_trip",
    )

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.staging-",
            dir=output_root.parent,
        ),
    )
    outputs = {
        "cohort_burden_source.csv": cohort_burden_source,
        "figure6_burden_source.csv": figure6_burden_source,
        "table_s5.csv": summary,
        "provider_overlap.csv": overlap,
        "top_primary_pairs.csv": top_pairs,
        "runtime_summary.csv": runtime,
    }
    for name, frame in outputs.items():
        frame.to_csv(staging / name, index=False, lineterminator="\n")

    display_columns = [
        "cohort",
        "tumors",
        "tested_pairs",
        "high_burden_fraction",
        "mutsig_primary_total",
        "mutsig_primary_me",
        "mutsig_primary_co",
        "cbase_primary_total",
        "dig_primary_total",
    ]
    latex = summary.loc[:, display_columns].to_latex(
        index=False,
        float_format="%.3f",
        escape=True,
    )
    (staging / "table_s5.tex").write_text(latex, encoding="utf-8")
    figure_path = staging / "figure6.pdf"
    _plot_figure6(
        burden_source=figure6_burden_source,
        summary=summary,
        calibration_table=calibration_table,
        primary_q=primary_q,
        output=figure_path,
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "contract": REPORT_CONTRACT,
        "cohorts": list(cohorts),
        "primary_provider": "mutsig",
        "primary_q_threshold": primary_q,
        "sensitivity_q_threshold": sensitivity_q,
        "high_burden_definition": {
            "measure": "pre-K total nonsynonymous SNV event count per tumor",
            "reference": "pooled 10,433-tumor 32-cohort analysis population",
            "pooled_tumor_count": EXPECTED_TUMOR_COUNT,
            "quantile": HIGH_BURDEN_QUANTILE,
            "threshold": burden_threshold,
            "comparison": "greater-than-or-equal",
            "interpretation": (
                "descriptive high-burden fraction, not a clinical hypermutator label"
            ),
        },
        "inputs": {
            "run_completion": _file_record(
                run_root / "completion_manifest.json",
                relative_to=run_root,
            ),
            "provider_manifest": _file_record(
                provider_root / "provider_manifest.json",
                relative_to=provider_root,
            ),
            "postprocess_manifest": _file_record(
                postprocess_root / postprocess.ROOT_MANIFEST_NAME,
                relative_to=postprocess_root,
            ),
            "calibration_summary": _file_record(
                calibration_root / calibration.SUMMARY_NAME,
                relative_to=calibration_root,
            ),
            "reporting_rule": {
                "path": rule_path.name,
                "bytes": rule_path.stat().st_size,
                "sha256": _sha256(rule_path),
            },
        },
        "outputs": {
            path.name: _file_record(path, relative_to=staging)
            for path in sorted(staging.iterdir())
            if path.is_file()
        },
    }
    _write_atomic(staging / "report_manifest.json", _canonical_json(manifest) + b"\n")
    staging.replace(output_root)
    validate_report(output_root)
    return output_root


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--provider-root", type=Path, required=True)
    parser.add_argument("--postprocess-root", type=Path, required=True)
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--reporting-rule", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main() -> None:
    """Build focused reporting artifacts from frozen inputs."""
    args = _parser().parse_args()
    print(
        build_report(
            run_root=args.run_root.resolve(),
            provider_root=args.provider_root.resolve(),
            postprocess_root=args.postprocess_root.resolve(),
            calibration_root=args.calibration_root.resolve(),
            rule_path=args.reporting_rule.resolve(),
            output_root=args.output_root.absolute(),
        ),
    )


if __name__ == "__main__":
    main()
