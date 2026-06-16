"""Per-cohort ME + CO interaction network plots for DIALECT (one figure per cohort).

For a cohort + BMR: apply the paper's eps-filter + intra-gene exclusion, take the
top-K mutually-exclusive pairs (rho<0, ranked by rho) and top-K co-occurring pairs
(rho>0, ranked by Likelihood Ratio), and draw a single network -- genes are nodes,
ME edges red, CO edges teal -- with bold prominent nodes and thick clear edges.

Usage::

    python analysis/plot_interaction_networks.py --cohort BRCA --bmr mutsig
    python analysis/plot_interaction_networks.py --cohort BRCA LUAD --bmr mutsig --top-k 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from analysis.bmr_passenger_proportions import rank_and_filter
from dialect.stats.thresholds import compute_epsilon_threshold

ME_COLOR = "#d62728"
CO_COLOR = "#17a2b8"


def _base(gene_effect: str) -> str:
    return gene_effect.rsplit("_", 1)[0]


def build_network(cohort: str, bmr: str, root: Path, top_k: int) -> tuple:
    """Return (graph, n_samples) with top-k ME + top-k CO edges for the cohort/BMR."""
    cohort_dir = root / cohort
    n = pd.read_csv(cohort_dir / "count_matrix.csv", index_col=0).shape[0]
    eps = compute_epsilon_threshold(n)
    df = pd.read_csv(cohort_dir / f"id_{bmr}" / "pairwise_interaction_results.csv")
    me = rank_and_filter(df, eps, "ME").head(top_k)
    co = rank_and_filter(df, eps, "CO").head(top_k)

    graph = nx.Graph()
    for _, r in me.iterrows():
        graph.add_edge(_base(r["Gene A"]), _base(r["Gene B"]), kind="ME",
                       weight=abs(r["Rho"]))
    for _, r in co.iterrows():
        a, b = _base(r["Gene A"]), _base(r["Gene B"])
        if graph.has_edge(a, b):  # a pair can't be both; ME wins by construction
            continue
        graph.add_edge(a, b, kind="CO", weight=abs(r["Rho"]))
    return graph, n


def draw(cohort: str, bmr: str, root: Path, out_dir: Path, top_k: int) -> Path | None:
    """Draw and save the ME/CO network for one cohort+BMR; None if no edges."""
    graph, n = build_network(cohort, bmr, root, top_k)
    if graph.number_of_edges() == 0:
        return None
    me_edges = [(u, v) for u, v, d in graph.edges(data=True) if d["kind"] == "ME"]
    co_edges = [(u, v) for u, v, d in graph.edges(data=True) if d["kind"] == "CO"]

    fig, ax = plt.subplots(figsize=(11, 9))
    pos = nx.spring_layout(graph, k=1.6, seed=1, iterations=200)
    nx.draw_networkx_nodes(graph, pos, node_size=2600, node_color="#ffe08a",
                           edgecolors="black", linewidths=2.5, ax=ax)
    nx.draw_networkx_edges(graph, pos, edgelist=me_edges, edge_color=ME_COLOR,
                           width=5.0, alpha=0.9, ax=ax)
    nx.draw_networkx_edges(graph, pos, edgelist=co_edges, edge_color=CO_COLOR,
                           width=5.0, alpha=0.9, ax=ax)
    nx.draw_networkx_labels(graph, pos, font_size=11, font_weight="bold", ax=ax)
    ax.plot([], [], color=ME_COLOR, lw=5, label=f"Mutual exclusivity (top {len(me_edges)})")
    ax.plot([], [], color=CO_COLOR, lw=5, label=f"Co-occurrence (top {len(co_edges)})")
    ax.legend(loc="upper left", fontsize=12, frameon=True)
    ax.set_title(f"{cohort} — DIALECT driver interactions ({bmr} BMR, N={n})",
                 fontsize=15, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    out = out_dir / f"{cohort}_{bmr}_network.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    """Draw ME/CO networks for the requested cohorts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", nargs="+", required=True)
    parser.add_argument("--bmr", default="mutsig")
    parser.add_argument("--results-root", default="output/pancan")
    parser.add_argument("--out-dir", default="figures/networks")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for cohort in args.cohort:
        saved = draw(cohort, args.bmr, Path(args.results_root), out_dir, args.top_k)
        print(f"  {cohort}: {saved or 'no edges (skipped)'}")


if __name__ == "__main__":
    main()
