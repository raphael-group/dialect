# AGENTS.md — DIALECT

PhD method: latent-variable model for **mutually exclusive (ME)** and **co-occurring (CO)** driver pairs. Each somatic count is `C = B + D` (passenger background + latent driver). The corrected revision uses one profile LRT for dependence and assigns ME/CO direction afterward from Marshall–Olkin `ρ`. **BMR is the load-bearing input.**

Public package: `raphael-group/dialect`. Companion atlas: `atlas/` (this repo).

## Goals & locks

- **Goal:** journal revision (rebuttal) + usable open-source tool. BMR is the highest-value technical workstream.
- **CO confound:** many CO calls are per-sample tumor-burden artifacts, not real co-drivers. Better **per-gene** BMR (DIG) does **not** fix CO; **per-sample** BMR (MutSig-style) does. ME biology largely holds.
- **Do not** propose burden-aware BMR switching (high-TMB vs low-TMB different BMR rules) — rejected as ad-hoc.
- **BMR framing:** one pluggable framework; sample-specific MutSigCV2 is the primary inferential background for both ME and CO; CBaSE is the continuity comparison, especially for ME; DIG is supplementary sensitivity. Provider overlap is descriptive, never a voting rule. This is one prospective hierarchy across cohorts and directions, not a burden-dependent switcher.
- **Revision family:** one matched, participant-unique K=500 universe under all three BMRs; exclude same-base `_M:_N` pairs before fitting; no epsilon prefilter and no probability floor or provider fallback.
- **Significance:** use one within-cohort family and one calibration-justified global p/q rule for both directions. Never choose a threshold to recover a desired cancer or pair.
- `research/` is **gitignored** (paper, notes, dissertation) — never commit it.
- Validate against **CHOL**; top ME should remain `IDH1_M : PBRM1_N`.

## Repo map

Layered DAG (enforced by `tests/test_architecture.py`):
`cli → api → (models | stats) → (bmr | baselines) → data` · `viz` + `experiments` at top.

| Path | Role |
|------|------|
| `src/dialect/api.py` | **Public seam** — `estimate_bmr`, `identify_interactions`, `compare_methods`, `merge_results` |
| `src/dialect/cli/` | Thin Typer wrappers over `api` |
| `src/dialect/models/` | EM core (`gene`, `interaction`, `assembly`) — pure math |
| `src/dialect/bmr/` | Pluggable `BMRProvider` (cbase, dig, …) |
| `src/dialect/data/` | Base layer: PMF/count I/O, `MutationCohort` |
| `src/dialect/baselines/` | Fisher / DISCOVER / MEGSA / WeSME |
| `src/dialect/stats/`, `viz/`, `experiments/` | Thresholds, ranking, plots, simulation |
| `src/dialect/utils/` | **Legacy re-export shims only** — import real homes |
| `external/` | Vendored CBaSE, DIG, MEGSA, WeSME (not importable packages) |
| `analysis/` | Paper figures / BMR sensitivity / atlas export |
| `tests/`, `docs/` | pytest + Sphinx |

## Data contract

- **`bmr_pmfs.csv`**: rows `GENE_M` / `GENE_N`; columns = integer counts; each row a PMF `P(B=k)` summing to 1.
- **`count_matrix.csv`**: samples × `GENE_M`/`GENE_N` integer somatic counts.
- Any `BMRProvider` must emit `bmr_pmfs.csv`-shaped output.

## Environment & commands

- Conda env **`dialect`** (Python 3.12): `/opt/anaconda3/envs/dialect/bin/python`
- Install: `pip install -e ".[dev]"` · Tests: `pytest` · Lint: `ruff check .`
- **Run from repo root.** `external/CBaSE/auxiliary/`, `data/`, `output/`, `research/` are gitignored.

```bash
dialect generate -m data/mafs/CHOL.maf -o output/CHOL          # needs CBaSE auxiliary
dialect identify -c output/CHOL/count_matrix.csv -b output/CHOL/bmr_pmfs.csv -o output/CHOL -k 100
```

```python
from dialect import estimate_bmr, identify_interactions
estimate_bmr("data/mafs/CHOL.maf", "output/CHOL", provider="cbase")
res = identify_interactions(
    "output/CHOL/count_matrix.csv", "output/CHOL/bmr_pmfs.csv", "output/CHOL", top_k=100
)
res.pairwise.sort_values("Rho").head()
```

Atlas release export (after deterministic baseline generation). Releases are
write-once; `--k` picks the contract and `--release-id` defaults to `k<K>-<UTC date>`.

```bash
# K=500 (current site default): the 32-cohort TCGA focused revision grid, frozen
# reporting rule (MutSigCV2 primary, BY q <= 0.01), columnar gzip tables.
PYTHONPATH=/path/to/DISCOVER/python \
  python -m analysis.build_atlas_baselines --profile k500 --jobs 6
python -m analysis.build_atlas_data --k 500 \
  --out atlas/public/data/releases/k500-<YYYY-MM-DD> \
  --release-id k500-<YYYY-MM-DD> --generated-at <YYYY-MM-DD>T00:00:00Z

# K=100 (historical, includes MSK): per-BMR top-100 layout, JSON tables.
PYTHONPATH=/path/to/DISCOVER/python \
  python -m analysis.build_atlas_baselines --profile k100 --jobs 4
python -m analysis.build_atlas_data --k 100 \
  --out atlas/public/data/releases/k100-2026-08-26 \
  --release-id k100-2026-08-26 --generated-at 2026-08-26T00:00:00Z

node atlas/scripts/validate-release.mjs   # validates every release on disk
```

DISCOVER must be the built package (it has a Fortran `_discover` extension):
`pip install --no-deps --target <dir> <DISCOVER>/python` at commit `a46d99f`.

## Gotchas

- BMR PMFs must sum to 1 (`load_bmr_pmfs` renormalizes + warns); count keys may be non-contiguous.
- EM excludes samples with no background support (hypermutators) — logs this; proper handling is open science work.
- `compare` DISCOVER/MEGSA need extra deps; they skip with a warning if absent.
- Prefer documented MutSig scripts over re-deriving the per-sample path.

## Style

src-layout, type hints, Google-style docstrings, `ruff` (`select = ["ALL"]`). Thin CLI; logic in `api` + layers. Prefer pure functions in `models/`. Test every change; `ruff check` clean before commit.
