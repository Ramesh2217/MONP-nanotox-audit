# Data

The benchmark dataset is not included in this repository.

This study uses the publicly available NanoTox metal-oxide nanoparticle cytotoxicity benchmark, released by Subramanian and Palaniappan (2021). To preserve dataset provenance and avoid redistributing third-party data, the benchmark is not included here and should be obtained directly from the original authors' repository.

## How to obtain the dataset

1. Download the original benchmark (dataset.txt) from the NanoTox repository: https://github.com/NanoTox/ToxicityModel
2. Place the downloaded file in this data/ folder.
3. Keep the name dataset.txt (no renaming needed).

The analysis scripts detect the file automatically via common.py. No code changes are required; just drop the file in data/ and run the scripts or notebooks.

## Source and license

Source: Subramanian, N.A., Palaniappan, A. (2021). NanoTox, ACS Omega 6, 11729-11739. https://doi.org/10.1021/acsomega.1c01076

Repository: https://github.com/NanoTox/ToxicityModel (MIT License; archived at Zenodo, https://doi.org/10.5281/zenodo.4055281). The benchmark draws on data from Choi et al. (2018) and Kar et al. (2014).
