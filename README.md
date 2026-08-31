# DIALECT

[![bioRxiv](https://img.shields.io/badge/bioRxiv-10.1101/2024.04.24.590995-olive)](https://www.biorxiv.org/content/10.1101/2024.04.24.590995v1)
[![License: BSD-3](https://img.shields.io/badge/License-BSD--3-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org)

**D**river **I**nteractions and **L**atent **E**xclusivity or **C**o-occurrence in **T**umors

DIALECT identifies mutually exclusive (ME) and co-occurring (CO) driver mutation pairs by modeling each somatic count as passenger background + latent driver, conditioned on a background mutation rate (BMR).

The commands below require a Git checkout installed in editable mode. This is
necessary for the default CBaSE provider: it resolves DIALECT's tracked CBaSE
fork under `external/CBaSE/`. The configured wheel contains the installable
`dialect` package, CLI metadata, and license. The configured source distribution
contains the source package, tests, and selected README, license, and build
metadata. Neither contains the `external/CBaSE` runtime scripts or auxiliary
data.

The checkout includes the CBaSE fork and its `NOTICE`, but its large
`external/CBaSE/auxiliary/` directory is intentionally not tracked. Before
running `dialect generate --bmr cbase`, provision a compatible auxiliary data
set there and review the
[`external/CBaSE/NOTICE`](external/CBaSE/NOTICE),
which records the upstream source and lineage caveat. DIALECT does not automate
that acquisition. If you already have `count_matrix.csv` and `bmr_pmfs.csv`, an
installed wheel can run `dialect identify` without CBaSE.

```bash
pip install -e .
dialect generate -m cohort.maf -o out/cohort
dialect identify -c out/cohort/count_matrix.csv -b out/cohort/bmr_pmfs.csv -o out/cohort -k 100
```

For development, install the test and lint dependencies with
`pip install -e ".[dev]"`.

For a zero-complete CBaSE cohort, provide the exact ordered sample axis as one
unique, nonempty identifier per line. The optional scalar is an equality assertion;
CBaSE's per-sample denominator is derived from the axis.

```bash
dialect generate -m cohort.maf -o out/cohort --bmr cbase \
  --cbase-sample-axis cohort_samples.txt --cbase-samples 137
```

```python
from dialect import estimate_bmr, identify_interactions
estimate_bmr(
    "cohort.maf",
    "out/cohort",
    provider="cbase",
    sample_ids="cohort_samples.txt",
    n_samples=137,
)
result = identify_interactions(
    "out/cohort/count_matrix.csv", "out/cohort/bmr_pmfs.csv", "out/cohort", top_k=100
)
```

Agent / contributor context: see [AGENTS.md](AGENTS.md).

## Cite

```bibtex
@article{shuaibi2024dialect,
  author  = {Ahmed Shuaibi and Uthsav Chitra and Benjamin J. Raphael},
  title   = {A latent variable model for evaluating mutual exclusivity and
             co-occurrence between driver mutations in cancer},
  journal = {bioRxiv},
  year    = {2024},
  doi     = {10.1101/2024.04.24.590995}
}
```

BSD-3-Clause — see [LICENSE](LICENSE).
