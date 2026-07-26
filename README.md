# Transformers for Asset Pricing

This repository implements the Dual Path Portfolio Transformer, based a cross sectional equity model trained directly on a Maximum Sharpe Ratio Regression objective on the JKP Emerging Markets factor panel.
 
## Model
 
The Dual Path Portfolio Transformer scores each firm for a six month holding horizon using two complementary paths. Path 1 scores every firm independently through a shared per firm head applied to market based (K0) and fundamentals (K1) characteristic blocks. Path 2 groups firms by country and, when the cross section is large enough, passes them through a small transformer with sparse top k attention, finally adding a learned adjustment to the Path 1 score.

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

> [!Note]
> There are five branches including the main branch in this repository. 
> - The `main` branch contains all the codebase of the final architecture after experimentation. 
> - `dualapproach` branch has a dual appraoch architecture
> -  `nonlinear/original` has the original kelly proposed architecture with different embedding varisnts.
> - The branch `nonlinear/time2vec` has a architecture involving Time2Vec encoding and periodic lag data. 
> - The `thesis\resources` contain diagrams for the dissertation.

> [!CAUTION]
> The data_processing notebook is run on a device with 32 GB ram. During the data process, the python has consumed more than ~29 GB data and complete 32 GB ram was utilised with ~10 GB swap memory as the complete dataset is loaded onto your ram. So, with that in mind, caution need be maintained when runnig the notebook on device with less resources.