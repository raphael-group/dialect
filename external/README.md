# External dependencies

This directory contains third-party source used by DIALECT. Inclusion in this working tree does not
imply that every dependency may be redistributed in a public release. The exact revision inventory,
license status, and include/exclude decisions are recorded under `provenance/dependencies/`.

## Integrated Codebases

### 1. CBaSE

- **Source**: [CBaSE v1.2 download page](https://genetics.bwh.harvard.edu/cbase/downloads_v1.2.html)
- **Version**: v1.2
- **Description**: CBaSE is a statistical framework designed for identifying positively selected driver mutations in cancer genomes.

`external/CBaSE/NOTICE` is the authoritative lineage, license, and modification notice. It
distinguishes the current official v1.2 reference archive from DIALECT's earlier two-script fork,
identifies the DIALECT-authored helper, and records the per-file Public Domain/BSD-3-Clause
boundary. The official archive and auxiliary data are not distributed by the DIALECT package.

#### Original Contributors:
- Donate Weghorn
- Shamil Sunyaev

#### Reference:
Weghorn, D., & Sunyaev, S. R. (2017). Bayesian inference of negative and positive selection in human cancers. *Nature Genetics*, 49(12), 1785–1788. [DOI](https://doi.org/10.1038/ng.3987)

---

### 2. DISCOVER (external acquisition)

- **Source**: [DISCOVER GitHub Repository](https://github.com/NKI-CCB/DISCOVER)
- **Version**: Python release 0.9.6, commit `a46d99f9a8a76dc6302f42c814650ca2a1568267`
- **License**: Apache-2.0
- **Description**: DISCOVER is a method for detecting mutual exclusivity and co-occurrence of genomic events in cancer data.

DISCOVER is acquired at the exact public commit for the corrected comparison workflow; its upstream
source archive is not tracked in this directory or copied into the revision deposit. The public
provenance record pins the source archive and license hashes.

#### Original Contributors:
- Sander Canisius
- John W. M. Martens
- Lodewyk F. A. Wessels

#### Reference:
Canisius, S., Martens, J. W. M., & Wessels, L. F. A. (2016). A novel independence test for somatic alterations in cancer shows that biology drives mutual exclusivity but chance explains most co-occurrence. *Genome Biology*, 17(1), 261. [Genome Biology Link](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-016-1114-x)

---

## Release boundary

MutSig2CV, MEGSA, DIG artifacts, OncoKB, and raw third-party data have separate restrictions or
unresolved terms. The public release verifier excludes those bytes and verifies a sanitized,
hash-pinned provenance record instead. The public comparison API retains optional WeSME support,
but no current corrected-revision dependency record or coauthor-approved comparator scope selects
WeSME or WeSCO. They therefore remain excluded from the prepared public release unless a future
stage-scoped decision explicitly selects them and a separate exact provenance, license, acquisition,
and redistribution review passes. Refer to each upstream project for lawful acquisition.
