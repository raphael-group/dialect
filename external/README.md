# External Codebases

This directory contains external codebases that have been integrated into this project to facilitate advanced analyses. Below are the details of each codebase, including the sources, modifications made, and references to the original contributors and publications.

## Integrated Codebases

### 1. CBaSE

- **Source**: [CBaSE v1.2 Download Page](http://genetics.bwh.harvard.edu/cbase/downloads_v1.2.html)
- **Version**: v1.2
- **Description**: CBaSE is a statistical framework designed for identifying positively selected driver mutations in cancer genomes.

#### Modifications Made:
1. Adjusted input and output (IO) file naming conventions to better align with the overall project structure.
2. Applied general formatting changes to adhere to the [Black](https://github.com/psf/black) code style for Python scripts:
   - `CBaSE_params_v1.2.py`
   - `CBaSE_qvals_v1.2.py`
3. Auxiliary files were used as-is without modifications.

#### Original Contributors:
- Donate Weghorn
- Shamil Sunyaev

#### Reference:
Weghorn, D., & Sunyaev, S. (2017). Bayesian inference of negative and positive selection in human cancers. *Genome Biology*, 18(1), 154. [PubMed Link](https://pubmed.ncbi.nlm.nih.gov/29106416/)

---

### 2. DISCOVER

- **Source**: [DISCOVER GitHub Repository](https://github.com/NKI-CCB/DISCOVER)
- **Version**: Python Release v0.9.5 ([GitHub Release Page](https://github.com/NKI-CCB/DISCOVER/releases/tag/py_v0.9.5))
- **Description**: DISCOVER is a method for detecting mutual exclusivity and co-occurrence of genomic events in cancer data.

#### Modifications Made:
1. Retained only the `python` directory files, as these are the only relevant components for integration.
2. Made minor changes to file input/output handling to ensure seamless integration with the overall project structure.

#### Original Contributors:
- Sander Canisius
- John W. M. Martens
- Lodewyk F. A. Wessels

#### Reference:
Canisius, S., Martens, J. W. M., & Wessels, L. F. A. (2016). A novel independence test for somatic alterations in cancer shows that biology drives mutual exclusivity but chance explains most co-occurrence. *Genome Biology*, 17(1), 261. [Genome Biology Link](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-016-1114-x)

---

## Notes on Usage
- These codebases have been modified for improved integration, particularly with regard to file naming conventions and Python script formatting.
- Any additional modifications made to these codebases in the future will be documented in this file.

For any issues regarding these external codebases, refer to their original sources or contact the contributors through their respective repositories or publications.
