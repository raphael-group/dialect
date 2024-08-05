## Installation

To install DIALECT, follow these steps:

## Installation
1. **Clone the Repository**:
   Clone the DIALECT repository from GitHub to your local machine using the following command:
   ```bash
   git clone https://github.com/raphael-group/dialect.git
   ```

2. **Create a Virtual Environment**:
   Navigate to the cloned directory and create a virtual environment:
   ```bash
   cd dialect
   python -m venv venv
   ```

3. **Activate the Virtual Environment**:
   - On Windows:
     ```bash
     .\venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install Dependencies**:
   Install the required dependencies using the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```

## Data

### Source

The primary data source for DIALECT is Mutation Annotation Format (MAF) files derived from the PanCancer Atlas, available through [cBioPortal for Cancer Genomics](https://www.cbioportal.org/datasets). These files contain comprehensive mutation data essential for analysis, including various types of mutations like missense, nonsense, frameshift, and synonymous mutations.

### Preparing TCGA Data

To prepare TCGA data for use with DIALECT, follow these detailed steps:

1. **Access Data**:
   - Navigate to the [cBioPortal's datasets search page](https://www.cbioportal.org/datasets).
   - Use the filter to select datasets marked as "TCGA, PanCancer Atlas".

2. **Download Data**:
   - Identify the dataset corresponding to the cancer subtype of interest.
   - Click on the download link to download the zipped data folder.

3. **Extract and Rename Files**:
   - Unzip the downloaded folder. It will typically be named something like `SUBTYPE_tcga_pan_can_atlas_2018`.
   - Within each unzipped folder, locate the file named `data_mutations.txt`.
   - Rename this file using its cancer subtype [abbreviation](https://gdc.cancer.gov/resources-tcga-users/tcga-code-tables/tcga-study-abbreviations) followed by a ".maf" suffix. For example, rename `data_mutations.txt` to `BRCA.maf` for Breast Cancer.

4. **Organize Data**:
   - Move all renamed `.maf` files to the `data/tcga_pancan_atlas` directory in your project folder. This directory is used by DIALECT to access mutation data for processing.

Alternatively, prepare your custom data with the proper MAF mutation data format.

### Preparing Custom Data

To prepare your custom mutation data for use with DIALECT:

1. **Format Data**:
   - Ensure your data is formatted as MAF. The required columns are: `CHROM`, `POS (1-based)`, `ID`, `REF`, `ALT`, `SAMPLE_ID`.

2. **Organize Data**:
   - Rename the mutation file to end with `.maf` and place it in the `data` directory. This standard naming and organization will allow DIALECT to easily access and process the data.

## Code Usage

DIALECT can be run from the core script `dialect/core.py`, which provides three separate subcommands. Each subcommand facilitates a different aspect of the analysis process:

- **generate**: This subcommand runs the [CBaSE method](https://dx.doi.org/10.1038/ng.3987) to generate background mutation rate (BMR) distributions from mutation data.
- **analyze**: This subcommand runs DIALECT to identify mutually exclusive and co-occurring interactions between genes.
- **compare**: This subcommand runs prior methods (including DISCOVER, Fisher's exact test, WeSME, WeSCO) to compare against DIALECT's results.

### Generate BMRs Using CBaSE

To generate BMRs using the CBaSE method, use the following command:
```bash
python dialect/core.py generate --method cbase data/tcga_pancan_atlas_2018/AML.maf results/tcga_pancan_atlas_2018/
```
This command processes the mutation data file `AML.maf` and outputs the BMR distributions into the specified results directory.

### Run DIALECT to Analyze Interactions

To analyze interactions between genes using DIALECT, use the following command:
```bash
python dialect/core.py analyze results/tcga_pancan_atlas_2018/AML/AML_cbase_cnt_mtx.csv results/tcga_pancan_atlas_2018/AML/AML_cbase_bmr_pmfs.csv results/tcga_pancan_atlas_2018/AML
```
This command analyzes the BMR distributions and mutation data to identify mutually exclusive and co-occurring interactions between genes.

### Run Comparison Methods

To run comparison methods and benchmark DIALECT against other tools, use the following command (to be completed):
```bash
# TODO
```

## License

DIALECT is distributed under the BSD-3 License. For more information, refer to the `LICENSE` file included in the repository.

## Contact

For any questions, issues, or contributions, please contact:
Ahmed Shuaibi - [ashuaibi@princeton.edu](mailto:ashuaibi@princeton.edu)

## Citation

If you use DIALECT in your research, please cite our paper:

- [Driver Interactions and Latent Exclusivity or Co-occurrence in Tumors](https://doi.org/10.1101/2024.04.24.590995)