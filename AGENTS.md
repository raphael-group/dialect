# AGENTS.md — working in the DIALECT repo

Context for AI agents (and humans) working on this codebase. Keep this file short and current.

## What DIALECT is

An EM latent-variable model that calls **mutually exclusive (ME)** and **co-occurring (CO)**
pairs of cancer **driver** mutations. Each observed somatic count is modeled as `C = B + D`:
a **passenger background** count `B` (from a background-mutation-rate model) plus a latent binary
**driver** indicator `D`. Per gene pair, the joint driver state `(D, D′)` follows a bivariate
Bernoulli `(τ₀₀, τ₀₁, τ₁₀, τ₁₁)` fit by EM; ME is ranked by Marshall–Olkin `ρ`, CO by a
likelihood ratio. The **background mutation rate (BMR)** is the load-bearing input.

## Repo map

The re-layout (`research/notes/09`) is **complete**: `utils/` was split into a one-way
layered DAG `cli → api → (models | stats) → (bmr | baselines) → data` (with `viz` +
`experiments` as top-of-stack orchestration). `tests/test_architecture.py` AST-enforces it.
The remaining `utils/*` files are re-export shims; new code should import the real homes.

- `src/dialect/api.py` — **the public seam.** `estimate_bmr`, `identify_interactions`,
  `compare_methods`, `merge_results` (the first two re-exported from the package root:
  `from dialect import estimate_bmr, identify_interactions`). The CLI and any agent/web
  backend call into this, not the internals.
- `src/dialect/cli/` — the **Typer** CLI. `app.py` = generate/identify/compare/merge +
  `--verbose`; `simulate.py` = the nested `simulate create|evaluate {single,pair,matrix}`.
  Thin wrappers over `api` (+ `experiments`). `__main__.py` delegates to `cli.app:main`.
- `src/dialect/models/` — the EM core (the science). `gene.py` = single-gene `π` EM;
  `interaction.py` = pairwise `τ` EM + `ρ`/LRT/Wald; `assembly.py` = build Gene/Interaction
  from a cohort (the data→models boundary). Pure math; treat with care.
- `src/dialect/bmr/` — the pluggable `BMRProvider` abstraction (imports only `data` + `bmr`).
  `base.py` = Protocol + `BMRResult`; `registry.py` = `get_provider`/`available`;
  `cbase.py`/`dig.py` = providers; `_cbase_run.py` = CBaSE subprocess; `_dig_pmf.py` = DIG math.
- `src/dialect/data/` — the dependency-free **base layer** (raw cancer data only). `io.py` =
  the data contract (PMF/count-matrix I/O); `cohort.py` = `MutationCohort` (counts + bmr_pmfs).
- `src/dialect/baselines/` — alternative ME/CO methods (`fishers`/`discover`/`megsa`/`wesme`)
  + `runner.py` (the `compare` dispatcher). Wrap `external/` via subprocess / guarded sys.path.
- `src/dialect/stats/` — `thresholds.py` (epsilon), `ranking.py` (top-ranked tables),
  `constants.py`. Pure statistics over result frames; never imports `viz`.
- `src/dialect/viz/` — `plotting.py` (all figures). `src/dialect/experiments/` — `simulate.py`
  (generation + evaluation + plotting orchestration; sits above `stats`+`viz`).
- `src/dialect/utils/` — **legacy shims only**: `identify.py` (EM orchestration, still here);
  `helpers.py`/`generate.py`/`dig_bmr.py`/`compare.py`/`fishers.py`/`discover.py`/`megsa.py`/
  `wesme.py`/`plotting.py`/`postprocessing.py`/`simulate.py` re-export their new homes.
  `argument_parser.py` still hosts `build_analysis_argument_parser` for `analysis/`.
- `external/` — vendored third-party tools (CBaSE, MEGSA, WeSME, DIGDriver). Invoked via wrappers,
  NOT shipped as an importable package. `external/CBaSE/auxiliary/` is gitignored (583 MB).
- `analysis/` — paper figure/table + `bmr_sensitivity.py` (CBaSE vs DIG). `tests/` — pytest
  (incl. `test_architecture.py`, which fails CI if the layering regresses). `docs/` — Sphinx.
- `research/` — **gitignored**; paper, dissertation, reviewer notes, and Claude context dossiers
  (`research/notes/00`–`10`). Never commit.

## The data contract

- **`bmr_pmfs.csv`**: rows `GENE_M` (missense) / `GENE_N` (nonsense), integer count columns; each
  row is a per-sample background PMF `P(B=k)` summing to 1. This is what every likelihood consumes.
- **`count_matrix.csv`**: samples × `GENE_M`/`GENE_N`, integer somatic counts.
- A `BMRProvider` (cbase/dig/dndscv) must emit `bmr_pmfs.csv`-shaped output regardless of its model.

## Environment & commands

- Dev env is the conda env **`dialect`** (Python 3.12): `/opt/anaconda3/envs/dialect/bin/python`.
  (Canonical reproducible env via **pixi** is planned — see blueprint.)
- Install: `pip install -e ".[dev]"`  ·  Tests: `pytest`  ·  Lint: `ruff check .`
- Run on the bundled CHOL test cohort, via the CLI:
  ```bash
  dialect generate -m data/mafs/CHOL.maf -o output/CHOL          # needs external/CBaSE/auxiliary
  dialect identify -c output/CHOL/count_matrix.csv -b output/CHOL/bmr_pmfs.csv -o output/CHOL -k 100
  ```
  …or as a library (the same code path the CLI uses):
  ```python
  from dialect import estimate_bmr, identify_interactions
  estimate_bmr("data/mafs/CHOL.maf", "output/CHOL", provider="cbase")
  res = identify_interactions("output/CHOL/count_matrix.csv",
                              "output/CHOL/bmr_pmfs.csv", "output/CHOL", top_k=100)
  res.pairwise.sort_values("Rho").head()   # strongest mutual exclusivity
  ```

## Invariants & gotchas

- **Run from the repo root** — CBaSE is invoked relative to it (now anchored via `__file__`, still
  prefer repo root). CBaSE runs under the same interpreter (`sys.executable`).
- `external/CBaSE/auxiliary/` and `data/`, `output/`, `research/` are **gitignored**; don't commit them.
- BMR PMFs must sum to 1 (`load_bmr_pmfs` renormalizes + warns); never assume contiguous count keys.
- The EM excludes samples with no background support (hypermutators) — it now logs this; the proper
  handling is an open science workstream.
- `compare`'s DISCOVER/MEGSA need extra deps (the `discover` pkg / R); they skip with a warning if absent.
- Validate every change against the CHOL run; top ME should remain `IDH1_M : PBRM1_N`.

## Style

src-layout, type hints, Google-style docstrings, `ruff` (`select = ["ALL"]`). Keep CLI handlers
thin — logic lives in `api.py` and the layers below it. Respect the one-way DAG (`test_architecture.py`
enforces `bmr → data`; extend its `ALLOWED_INTERNAL_PREFIXES` as each layer is cleaned). Prefer pure
functions in `models/`. Add a test with every change; verify `ruff check` is clean **before** committing.
