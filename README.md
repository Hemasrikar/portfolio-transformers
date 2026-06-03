## Data Processing

This branch contains the code base for all the data processing needed from this project.

All the data is obtained from WRDS (Wharton Research Data Services).

The [Company Data csv](csv_data\company_data_info.csv) file all the linking data from Compustat and CRSP. The datasets are seperated in three ways: Emerging Markets, USA market, and all the markets in the world. Primarily, we will work with the EM dataset.

### Train, Test, Validation Split

All the data split will be done prior to the data processing, and it is done based on the time period. 
> Train Period: 1995 - 2015
> Validation Period: 2016 - 2020
> Test Period: 2021 - 2025

The features are removed, if their respective column has more than 30% missing data. This is processed on train dataset and the kept features are used to filter the data in the validation and test datasets. After filtering the features, the train dataset will be processed. 