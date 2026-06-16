"""Top literature-corroborated CO clusters DIALECT recovers (rebuttal figure).

Each panel is a known, named co-occurrence module. Edge style encodes
BMR-robustness: solid+thick = m-robust (CO survives the per-sample MutSig2CV
BMR -> not a burden artifact); thin+dashed = recovered only under per-gene
BMRs (CBaSE/DIG). Node = driver gene.

Usage:  python -m analysis.plot_co_clusters
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

CO = "#0f9d9d"
OUT = Path("figures/co_clusters_corroborated.png")

# (title, subtitle, edges[(a, b, established?, m_robust?)])
MODULES = [
    ("Pancreatic — 4-driver progression module",
     "KRAS · TP53 · SMAD4 · CDKN2A   (KRAS:TP53 replicated in 3 studies)",
     [("KRAS", "TP53", True, True), ("KRAS", "CDKN2A", True, True),
      ("KRAS", "SMAD4", True, True), ("CDKN2A", "TP53", True, True),
      ("SMAD4", "TP53", True, False)]),
    ("Glioma — IDH-mutant astrocytoma triad + oligodendroglioma arm",
     "astrocytoma IDH1:TP53:ATRX  ·  oligodendroglioma CIC:FUBP1:IDH",
     [("IDH1", "TP53", True, True), ("ATRX", "TP53", True, True),
      ("ATRX", "IDH1", True, True), ("CIC", "IDH1", True, False),
      ("CIC", "FUBP1", True, False), ("FUBP1", "IDH1", True, False),
      ("CIC", "IDH2", True, False)]),
    ("Lung NSCLC — KL / KRAS immunotherapy-resistant subtype",
     "KRAS:STK11:KEAP1   (both core pairs replicated in 2 studies)",
     [("KRAS", "STK11", True, True), ("KEAP1", "STK11", True, True),
      ("KRAS", "U2AF1", False, False)]),
    ("Ovarian — clear-cell / endometrioid module",
     "ARID1A:PIK3CA experimentally validated (Chandler 2015)",
     [("ARID1A", "PIK3CA", True, True), ("CTNNB1", "PIK3CA", True, False),
      ("ARID1A", "CTNNB1", True, False), ("ARID1A", "PTEN", True, False)]),
]


def draw(ax: plt.Axes, title: str, subtitle: str, edges: list[tuple]) -> None:
    """Draw one CO-module network panel (thick=m-robust, dashed=per-gene-only)."""
    g = nx.Graph()
    for a, b, est, m in edges:
        g.add_edge(a, b, est=est, m=m)
    pos = nx.spring_layout(g, k=1.4, seed=3, iterations=300)
    nx.draw_networkx_nodes(g, pos, node_size=2400, node_color="#ffe08a",
                           edgecolors="black", linewidths=2.2, ax=ax)
    nx.draw_networkx_labels(g, pos, font_size=11, font_weight="bold", ax=ax)
    for a, b, _est, m in edges:
        nx.draw_networkx_edges(
            g, pos, edgelist=[(a, b)], edge_color=CO,
            width=6.0 if m else 2.5, style="solid" if m else (0, (4, 3)),
            alpha=0.95 if m else 0.7, ax=ax)
    ax.set_title(title, fontsize=12.5, fontweight="bold", loc="left")
    ax.set_xlabel(subtitle, fontsize=9.5, color="#444")
    ax.axis("off")
    ax.margins(0.18)


def main() -> None:
    """Render the four-panel corroborated-CO-modules figure to OUT."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    for ax, (t, s, e) in zip(axes.ravel(), MODULES, strict=False):
        draw(ax, t, s, e)
    m_lbl = "m-robust CO  (survives per-sample MutSig2CV — not a burden artifact)"
    g_lbl = "CO recovered under per-gene BMR only (CBaSE/DIG)"
    fig.legend(handles=[
        plt.Line2D([], [], color=CO, lw=6, label=m_lbl),
        plt.Line2D([], [], color=CO, lw=2.5, ls=(0, (4, 3)), label=g_lbl),
    ], loc="lower center", ncol=2, fontsize=11, frameon=True,
        bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Literature-corroborated co-occurrence modules DIALECT recovers",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
