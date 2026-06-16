"""Two-panel figure for the burden-aware BMR story (rebuttal note 14).

Panel A: per-cohort #CO pairs vs median TMB for the three BMRs (log axes).
Per-sample MutSig collapses burden-driven spurious CO at high TMB but
over-corrects (to ~0) at low TMB; per-gene CBaSE/DIG keep real CO at low TMB.
Panel B: AML (LAML) per-driver observed/expected-background ratio under
per-sample MutSig. DNMT3A (obs/exp~1.35) is explained away as passenger
background -- the mechanism of the low-TMB over-correction.

Usage:  python -m analysis.plot_burden_aware
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

from analysis.bmr_overcorrection_check import laml_lambda, tmb_co

OUT = Path("figures/bmr_burden_aware.png")
COLORS = {"cbase": "#7f7f7f", "dig": "#1f77b4", "mutsig": "#d62728"}
LABELS = {"cbase": "CBaSE (per-gene)", "dig": "DIG (per-gene)",
          "mutsig": "MutSig2CV (per-sample)"}


def main() -> None:
    """Render the two-panel burden-aware BMR figure to OUT."""
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(16, 6.5))

    # ---- Panel A: #CO vs median TMB ----
    t = tmb_co().dropna(subset=["medTMB"]).copy()
    for bmr in ("cbase", "dig", "mutsig"):
        y = t[f"CO_{bmr}"].clip(lower=0.5)  # 0 -> 0.5 so it shows on log axis
        ax_a.scatter(t["medTMB"], y, s=70, color=COLORS[bmr], label=LABELS[bmr],
                     alpha=0.85, edgecolors="white", linewidths=0.8, zorder=3)
    ax_a.axvspan(6, 30, color="#fff2cc", alpha=0.6, zorder=0)
    ax_a.text(13, 0.62, "low-TMB:\nMutSig over-corrects", fontsize=10,
              ha="center", color="#8a6d00", fontweight="bold")
    ax_a.axvspan(80, 450, color="#e8f0fe", alpha=0.7, zorder=0)
    ax_a.text(190, 0.62, "high-TMB:\nMutSig correctly collapses\nspurious CO",
              fontsize=10, ha="center", color="#1a4d8f", fontweight="bold")
    # annotate a few telling cohorts
    for c, dy in [("LAML", 1.1), ("PAAD", 1.1), ("PRAD", 1.1),
                  ("BRCA", 1.15), ("CRAD", 1.15), ("SKCM", 1.15), ("UCS", 1.1)]:
        row = t[t["cohort"] == c]
        if not row.empty:
            ax_a.annotate(c, (row["medTMB"].iloc[0],
                              max(row["CO_cbase"].iloc[0], 0.5) * dy),
                          fontsize=8.5, ha="center", color="#333")
    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xlabel("Cohort median tumor mutation burden (events / sample)",
                    fontsize=12)
    ax_a.set_ylabel("# co-occurring pairs (eps-filter + intra-gene excl.)",
                    fontsize=12)
    ax_a.set_title(
        "A  Per-sample BMR: corrective at high TMB, over-corrective at low TMB",
        fontsize=13, fontweight="bold", loc="left")
    ax_a.legend(fontsize=10, loc="upper left", framealpha=0.95)
    ax_a.grid(visible=True, which="both", alpha=0.25)

    # ---- Panel B: LAML obs/exp per AML driver ----
    df = laml_lambda().copy()
    df = df[df["obs_total"] >= 3]                      # drop ultra-sparse effects
    df = df.sort_values("obs_over_exp")
    colors = ["#d62728" if r < 2 else "#9ecae1" for r in df["obs_over_exp"]]
    bars = ax_b.barh(df["gene_effect"], df["obs_over_exp"],
                     color=colors, edgecolor="black", linewidth=0.6)
    ax_b.axvline(1.0, color="black", ls="--", lw=1.2)
    ax_b.text(1.05, 0.2, "obs = background\n(driver explained away)", fontsize=9,
              color="#d62728", fontweight="bold")
    for b, v in zip(bars, df["obs_over_exp"], strict=False):
        ax_b.text(v * 1.05, b.get_y() + b.get_height() / 2, f"{v:.1f}",
                  va="center", fontsize=8.5)
    ax_b.set_xscale("log")
    ax_b.set_xlabel("observed / expected-background  (per-sample MutSig, LAML)",
                    fontsize=12)
    ax_b.set_title("B  AML: only DNMT3A's background is inflated (obs/exp ~ 1)",
                   fontsize=13, fontweight="bold", loc="left")
    ax_b.grid(visible=True, axis="x", which="both", alpha=0.25)

    fig.suptitle(
        "Burden-aware BMR selection: match the background to the burden regime",
        fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
