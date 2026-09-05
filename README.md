# Transformers for Asset Pricing

![Status](https://img.shields.io/badge/status-complete-brightgreen)

This project was part of my MSc dissertation and was conducted in association with Rothko Investment Strategies. I developed a **Dual Path Portfolio Transformer (DPPT)**, a cross-sectional equity model trained directly using a **Maximum Sharpe Ratio Regression objective** on the [JKP Emerging Markets](https://jkpfactors.com) factor panel.
 
## Model
 
The Dual Path Portfolio Transformer (DPPT) scores each firm for a **six-month holding horizon** using two complementary paths. Path 1 scores each firm independently through a shared per-firm head applied to market-based (**K0**) and fundamental (**K1**) characteristic blocks. Path 2 groups firms by country and, when the cross-section is sufficiently large, passes them through a compact transformer with **sparse top-k attention**, producing a learned adjustment that is added to the Path 1 score.

---

### Train, Test, Validation Split

All the data split will be done prior to the data processing, and it is done based on the time period. 
> Train Period: 1995 - 2015
> Validation Period: 2016 - 2020
> Test Period: 2021 - 2025

The features are removed, if their respective column has more than 30% missing data. This is processed on train dataset and the kept features are used to filter the data in the validation and test datasets. After filtering the features, the train dataset will be processed. 

## Codebase
 
`src/` contains the core pipeline.
 
* `data_processing.py`, For data processing.
* `nonlinear_dualapproach.py`, the main training script. Run preprocessing when needed and trains all five encoder variants across multiple random seeds.
* `hpt_dual_path.py`, an Optuna based hyperparameter search over the same five variants, using validation set with long short Sharpe ratio as the objective.
* `eval_dual_path.py`, the evaluation script, which ensembles trained seeds per variant and reports long only, long short, and country composite portfolio performance at six month and monthly rebalance frequencies.
* `score_orthogonality_test.py`, a Fama Macbeth test of whether Path 2 adds information beyond Path 1, with Newey West standard errors.
`benchmark/` holds the comparison models. A Fama French five factor benchmark, an XGBoost and LightGBM benchmark, an MLP benchmark, and a Feature Tokeniser Transformer benchmark with its own tuning script, sharing common portfolio simulation and metrics utilities in `benchmark_common.py`.
 
`notebooks/` contains country level analysis and summary plotting notebooks. `slurm/` holds the job scripts used to run training, tuning, and evaluation on a GPU cluster, with corresponding logs saved in `job-output/`. `plots/` and `results/` hold generated figures and metrics for each model.

> The final HPC output files in the `job-output` are
> - Transformer Hyperparameter Tunning: 1241896
> - Transformer Training: 1241897
> - Transformer Evaluation: 1241898
> - Transformer Analysis: 1233650

> [!Note]
> There are five branches including the main branch in this repository. 
> - The `main` branch contains all the codebase of the final architecture after experimentation. 
> - `dualapproach` branch has a dual appraoch architecture
> -  `nonlinear/original` has the original kelly proposed architecture with different embedding varisnts.
> - The branch `nonlinear/time2vec` has a architecture involving Time2Vec encoding and periodic lag data. 
> - The `thesis\resources` contain diagrams for the dissertation.

> [!CAUTION]
> The data_processing script is run on a device with 32 GB ram. During the data processing, the python has consumed more than ~29 GB ram with ~10 GB swap memory.

---
## Transformer Architecture

![Dual Path Portfolio Transformer](https://github.com/Hemasrikar/portfolio-transformers/blob/thesis/resources/dualapproach-transformer.png)
