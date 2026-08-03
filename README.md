# DIALECT

[![bioRxiv](https://img.shields.io/badge/bioRxiv-10.1101/2024.04.24.590995-olive)](https://www.biorxiv.org/content/10.1101/2024.04.24.590995v1)
[![License: BSD-3](https://img.shields.io/badge/License-BSD--3-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org)

**D**river **I**nteractions and **L**atent **E**xclusivity or **C**o-occurrence in **T**umors

DIALECT identifies mutually exclusive (ME) and co-occurring (CO) driver mutation pairs by modeling each somatic count as passenger background + latent driver, conditioned on a background mutation rate (BMR).

```bash
pip install -e ".[dev]"
dialect generate -m cohort.maf -o out/cohort
dialect identify -c out/cohort/count_matrix.csv -b out/cohort/bmr_pmfs.csv -o out/cohort -k 100
```

```python
from dialect import estimate_bmr, identify_interactions
estimate_bmr("cohort.maf", "out/cohort", provider="cbase")
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
