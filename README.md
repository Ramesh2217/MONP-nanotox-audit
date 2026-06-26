# MONP-nanotox-audit

Validation audit of a metal-oxide nanoparticle cytotoxicity benchmark.

Full analysis pipeline for the study "Validation strategy and duplicate records shape apparent machine-learning performance in a metal-oxide nanoparticle cytotoxicity benchmark." The code reproduces every reported number and figure from the raw benchmark.

Author: A. Ramesh Babu (ORCID: https://orcid.org/0000-0002-8173-2005)

Data: Uses the public NanoTox benchmark (Subramanian and Palaniappan, 2021) from https://github.com/NanoTox/ToxicityModel. The dataset is not redistributed here; download it and place it in the data folder before running.

To reproduce: install dependencies with pip install -r requirements.txt, place the dataset in the data folder, then run python run_all.py. The pipeline ends with a self-check confirming 364 unique records (68 cytotoxic). See REPRODUCE.md for details.

License: MIT (see LICENSE). The dataset retains its original NanoTox license.
