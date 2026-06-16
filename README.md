# Nonlinear Transformer for Asset Pricing
This branch contains codebase of nonlinear transformer with Time2Vec encoding and factors split (K0/K1) implementation. This is the model that is proposed in the dissertation proposal.

The code base has the data processing code included in it. So, it can be run standalone on the raw data set. The traning is done on the dataset after split as mentioned below. 

The [model documentation](model_documentation.md) provided detailed implemention of the codebase.

---

### Train, Test, Validation Split

All the data split will be done prior to the data processing, and it is done based on the time period. 
> Train Period: 1995 - 2015
> Validation Period: 2016 - 2020
> Test Period: 2021 - 2025

The features are removed, if their respective column has more than 30% missing data. This is processed on train dataset and the kept features are used to filter the data in the validation and test datasets. After filtering the features, the train dataset will be processed. 


> [!CAUTION]
> The data_processing notebook is run on a device with 32 GB ram. During the data process, the python has consumed more than ~29 GB data and complete 32 GB ram was utilised with ~10 GB swap memory as the complete dataset is loaded onto your ram. So, with that in mind, caution need be maintained when runnig the notebook on device with less resources.

> [!Note]
> There are five branches including the main branch in this repository. 
> - The `main` branch contains all the codebase of the final architecture after experimentation. 
> - `dualapproach` branch has different a dual appraoch architecture with MLP layers before the embedding layer, and other additional statistical embedding.
> -  `nonlinear/original` has the original kelly proposed architecture with different embedding varisnts.
> - The branch `nonlinear/time2vec` has a architecture involving Time2Vec encoding and periodic lag data. 
> - The `thesis\resources` contain diagrams for the dissertation.

