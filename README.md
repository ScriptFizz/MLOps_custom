# MLOps regression project with MLFlow

In this project we tackle a regression problem, i.e. the prediction of housing price in King County, USA (datasets available on <a href=https://www.kaggle.com/datasets/harlfoxem/housesalesprediction>Kaggle</a>) in a CI-CD pipeline.

## Tools used

- Python
- DVC
- MLFlow

## Project structure

The code for the ML pipeline is stored in the src folder:

``` bash

src
├── __init__.py
├── data
│   ├── __init__.py
│   ├── loading_data.py
│   ├── split_data.py
│   └── transform_data.py
├── models
│   ├── production_model_selection.py
│   └── training_pipeline.py
├── prediction
├── steps
│   ├── __init__.py
│   ├── cleaning_data.py
│   ├── evaluation.py
│   ├── ingest_data.py
│   └── model_train.py
└── utils
    ├── __init__.py
    ├── data_cleaning.py
    ├── evaluation.py
    ├── methods.py
    └── model_development.py


```
