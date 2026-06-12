# 🧬 DIALECT

[![bioRxiv](https://img.shields.io/badge/bioRxiv-10.1101/2024.04.24.590995-olive)](https://www.biorxiv.org/content/10.1101/2024.04.24.590995v1)
[![License: BSD-3](https://img.shields.io/badge/License-BSD--3-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**D**river **I**nteractions and **L**atent **E**xclusivity or **C**o-occurrence in **T**umors

DIALECT identifies **mutually exclusive (ME)** and **co-occurring (CO)** pairs of driver
mutations in cancer. Unlike methods that binarize the mutation matrix, DIALECT models each
observed somatic count as a latent sum of a **passenger background** (`B`) plus a **driver
indicator** (`D`), and fits the joint driver state of each gene pair by EM. By conditioning on a
per-gene **background mutation rate (BMR)**, it suppresses the spurious dependencies that long or
highly-mutated genes (e.g. *TTN*) induce in binarization-based tests.

## Quickstart

```bash
# install (editable, with dev tooling)
pip install -e ".[dev]"

# 1. generate background PMFs + count matrix from a MAF (runs the BMR provider)
dialect generate -m cohort.maf -o out/cohort

# 2. identify mutually exclusive / co-occurring driver pairs (the EM)
dialect identify -c out/cohort/count_matrix.csv -b out/cohort/bmr_pmfs.csv -o out/cohort -k 100

# 3. (optional) benchmark against Fisher / DISCOVER / MEGSA / WeSME
dialect compare -c out/cohort/count_matrix.csv -o out/cohort -k 100
```

```python
# the same logic is available as a typed Python API (notebooks, pipelines, web)
import dialect as dl
cohort  = dl.read_maf("cohort.maf", bmr="cbase", reference="hg19")
results = dl.identify(cohort, top_k=100)   # -> pandas.DataFrame of ME/CO pairs
```

> **Background mutation rate is pluggable.** DIALECT supports multiple BMR providers
> (`cbase`, `dig`, ...) behind one interface, so you can benchmark how robust your ME/CO calls
> are to the choice of background model.

See the [documentation](https://github.com/raphael-group/dialect) for tutorials, the methods
write-up, and the API reference.

## Citing DIALECT

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

## License

BSD-3-Clause — see [LICENSE](LICENSE).

📧 [ashuaibi@princeton.edu](mailto:ashuaibi@princeton.edu)
