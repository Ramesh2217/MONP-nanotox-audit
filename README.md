MONP-nanotox-audit
Validation audit of a metal-oxide nanoparticle cytotoxicity benchmark

MONP-nanotox-audit
Validation audit of a metal-oxide nanoparticle cytotoxicity benchmark.

This repository contains the full analysis pipeline for the study "Validation strategy and duplicate records shape apparent machine-learning performance in a metal-oxide nanoparticle cytotoxicity benchmark." The code reproduces every reported number and figure from the raw benchmark in a single run.

Author
A. Ramesh Babu — ORCID: https://orcid.org/0000-0002-8173-2005

Data
The analysis uses the publicly available NanoTox benchmark (Subramanian & Palaniappan, 2021), obtained from https://github.com/NanoTox/ToxicityModel. The dataset is not redistributed here; download dataset.txt from the NanoTox repository and place it in the data/ folder before running. See data/README.md for details.

Reproducing the analysis
Install dependencies: pip install -r requirements.txt
Place the dataset in data/ (see above).
Run the full pipeline: python run_all.py
The pipeline runs all analysis stages in order and ends with a self-check confirming the dataset reduces to 364 unique records (68 cytotoxic). See REPRODUCE.md for a step-by-step walkthrough.

License
Code is released under the MIT License (see LICENSE). The underlying dataset retains its original NanoTox license and provenance.
