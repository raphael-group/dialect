# CBaSE Reference Files

A description of the folders, their respective files, and the file sources.

## Auxiliary

Additional data files and scripts provided with [CBaSE method](https://dx.doi.org/10.1038/ng.3987). Latest download for publicly available CBaSE scripts & files available [here](http://genetics.bwh.harvard.edu/cbase/downloads.html). At the time of writing, the latest scripts do not incorporate sample specific background mutation rate distributions; direct modifications to the method were made by Donate Weghorn's group for the sake of obtaining the values for Raphael's group.

## MAFS

MAFs are from [PanCanAtlas](https://gdc.cancer.gov/about-data/publications/pancanatlas). Specific data files were downloaded from [cBioPortal](https://www.cbioportal.org/datasets). Data retrieval instructions:
1. Navigate to [cBioPortal's datasets search page](https://www.cbioportal.org/datasets).
2. Filter dataset search with "TCGA, PanCancer Atlas".
3. Download zipped folder for each subtype maf & unzip folders.
4. Locate file named "data\_mutations.txt" for each cancer subtype. This file is the subtype MAF.
5. Rename the "data\_mutations.txt" file to its [abbreviation](https://gdc.cancer.gov/resources-tcga-users/tcga-code-tables/tcga-study-abbreviations) and use a ".maf" suffix to get files named like "BRCA.maf".
6. Place all subtype mafs in "/reference/cbase/mafs" directory.
