# Data

**The benchmark dataset is not included in this repository.**

This study uses the publicly available NanoTox metal-oxide nanoparticle
cytotoxicity benchmark, released by Subramanian and Palaniappan (2021). To
preserve dataset provenance and avoid redistributing third-party data, the
benchmark is not included here and should be obtained directly from the original
authors' repository.

## How to obtain the dataset

1. Download the original benchmark (`dataset.txt`) from the NanoTox repository:
   - File: https://github.com/NanoTox/ToxicityModel/blob/master/dataset.txt
   - Raw: https://raw.githubusercontent.com/NanoTox/ToxicityModel/master/dataset.txt
2. Place the downloaded file in **this** `data/` folder.
3. Keep the name `dataset.txt` (no renaming needed).

The analysis scripts detect the file automatically. The loader (`scripts/common.py`)
will use any one of the following, in this order, if present in `data/`:

```
data/dataset.txt                 (tab-separated, as downloaded — recommended)
data/NanoTox_dataset_clean.xlsx  (Excel version, if you have one)
data/NanoTox.csv                 (comma-separated version, if you have one)
```

No code changes are required; just drop the file in `data/` and run the scripts
or notebooks.

## About the file

`dataset.txt` is a tab-separated text file with a header row and 483 data rows
describing five metal-oxide nanoparticles (Al2O3, CuO, Fe2O3, TiO2, ZnO). It has
24 columns: an oxide identifier (`NPs`), physicochemical and periodic-property
descriptors, exposure conditions, a continuous `viability` readout, and a binary
`class` label (`Toxic` / `nonToxic`, thresholded at <50% viability).

The analysis selects nine descriptors (core size, hydrodynamic size, surface
charge, surface area, conduction band energy `Ec`, exposure time, dose,
electronegativity `e`, and number of oxygen atoms `NOxygen`) and derives a
deduplicated set of 364 unique records (68 cytotoxic) via the procedure in
`scripts/duplicate_audit.py` (see also `scripts/common.py`).

## Source and license

- Source: Subramanian, N.A., Palaniappan, A. (2021). NanoTox: development of a
  parsimonious in silico model for toxicity assessment of metal-oxide
  nanoparticles using physicochemical features. *ACS Omega* 6, 11729-11739.
  https://doi.org/10.1021/acsomega.1c01076
- Repository: https://github.com/NanoTox/ToxicityModel (MIT License;
  archived at Zenodo, https://doi.org/10.5281/zenodo.4055281)
- The benchmark in turn draws on toxicity data compiled by Choi et al. (2018)
  and descriptor information reported by Kar et al. (2014).
