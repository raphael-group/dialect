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

- `src/dialect/models/` — the EM core (the science). `gene.py` = single-gene `π` EM;
  `interaction.py` = pairwise `τ` EM + `ρ`/LRT/Wald. Pure math; treat with care.
- `src/dialect/utils/` — pipeline glue (being re-laid-out into `data/ bmr/ baselines/ stats/
  viz/ cli/` — see `research/notes/08_repo_reinvention_blueprint.md`). Notable:
  - `generate.py` — MAF → CBaSE → `bmr_pmfs.csv` + count matrix.
  - `identify.py` — runs the EM, writes `single_gene_results.csv` / `pairwise_interaction_results.csv`.
  - `dig_bmr.py` — DIG → DIALECT BMR adapter (prototype for the `BMRProvider` abstraction).
  - `compare.py` — benchmark vs Fisher/DISCOVER/MEGSA/WeSME (each method isolated; missing deps skip).
  - `helpers.py` — `load_bmr_pmfs`, `initialize_gene_objects`, etc.
- `external/` — vendored third-party tools (CBaSE, MEGSA, WeSME, DIGDriver). Invoked via wrappers,
  NOT shipped as an importable package. `external/CBaSE/auxiliary/` is gitignored (583 MB).
- `analysis/` — paper figure/table scripts. `tests/` — pytest. `docs/` — Sphinx.
- `research/` — **gitignored**; paper, dissertation, reviewer notes, and Claude context dossiers
  (`research/notes/00`–`08`). Never commit.

## The data contract

- **`bmr_pmfs.csv`**: rows `GENE_M` (missense) / `GENE_N` (nonsense), integer count columns; each
  row is a per-sample background PMF `P(B=k)` summing to 1. This is what every likelihood consumes.
- **`count_matrix.csv`**: samples × `GENE_M`/`GENE_N`, integer somatic counts.
- A `BMRProvider` (cbase/dig/dndscv) must emit `bmr_pmfs.csv`-shaped output regardless of its model.

## Environment & commands

- Dev env is the conda env **`dialect`** (Python 3.12): `/opt/anaconda3/envs/dialect/bin/python`.
  (Canonical reproducible env via **pixi** is planned — see blueprint.)
- Install: `pip install -e ".[dev]"`  ·  Tests: `pytest`  ·  Lint: `ruff check .`
- Run on the bundled CHOL test cohort:
  ```bash
  dialect generate -m data/mafs/CHOL.maf -o output/CHOL          # needs external/CBaSE/auxiliary
  dialect identify -c output/CHOL/count_matrix.csv -b output/CHOL/bmr_pmfs.csv -o output/CHOL -k 100
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

src-layout, type hints, NumPy-style docstrings, `ruff` (`select = ["ALL"]`). Keep CLI handlers thin
(logic lives in the API/core). Prefer pure functions in `models/`. Add a test with every change.
