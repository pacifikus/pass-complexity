# pass complexity

[![Tests](https://github.com/pacifikus/pass-complexity/actions/workflows/tests.yml/badge.svg)](https://github.com/pacifikus/pass-complexity/actions/workflows/tests.yml)
[![Code review](https://github.com/pacifikus/pass-complexity/actions/workflows/code-review.yml/badge.svg)](https://github.com/pacifikus/pass-complexity/actions/workflows/code-review.yml)
[![wemake-python-styleguide](https://img.shields.io/badge/style-wemake-000000.svg)](https://github.com/wemake-services/wemake-python-styleguide)


## Overview

This is a project to predict password complexity. Target is a real value indicating how many times the password could be encountered in one million random passwords.

## Data

Data were taken from the Kaggle competition [DMIA.ProductionML 2021.1 Password complexity](https://www.kaggle.com/competitions/dmia-production-ml-2021-1-passwords/overview)

## Metrics

Main metric is RMSLE.  RMSLE is preferable when

- targets having exponential growth, such as population counts, average sales of a commodity over a span of years etc
- we care about percentage errors rather than the absolute value of errors.
- there is a wide range in the target variables and we don’t want to penalize big differences when both the predicted and the actual are big numbers.
- we want to penalize under estimates more than over estimates.

You can find more information [here](https://hrngok.github.io/posts/metrics/#Root-Mean-Squared-Logaritmic-Error-(RMSLE))

## Experiments setup

- Hardware
    - CPU count: 1
    - GPU count: 1
    - GPU type: Tesla T4
- Software:
    - Python version: 3.7.14
    - OS: Linux-5.10.133+-x86_64-with-Ubuntu-18.04-bionic

| Model                                 | RMSLE  |
|---------------------------------------|--------|
| Vanilla Linear Regression             | 0.5023 | 
| Random Forest Regressor               | 0.5018 | 
| RF Regressor with hyperparams tuning  | 0.5000 | 
| TF-IDF + Linear Regression            | 0.4831 | 
| Custom LSTM                           | 0.3428 | 

## Recommended production hardware requirements

- Hardware
    - CPU: 4 CPU Cores
    - GPU: single GPU with at least 4 GB GPU RAM (btw, you can use only CPU model inference. See also [CPU inference optimization](https://youtu.be/okcvDWkyw2Y?t=23964))
    - RAM: 8 GB
    - System disk space: 2 GB
 
## How to run

### Install dependencies

First of all, install project dependencies, with command:

```
pip install -r src/requirements.txt
```

### Pipelines

The project includes 3 pipelines:
- data processing pipeline with etl function
- data science pipelines with a train-test split and a model fitting
- inference pipeline with a prediction for the test data

Current pipeline DAG looks like:

![Pipeline DAG](/imgs/pipeline_dark.png)


### How to run pipelines

You can run pipelines from the project with:

```
kedro run
```

### How to run app

You can run served model with [Flask](https://flask.palletsprojects.com/en/2.2.x/) and [Waitress WSGI server](https://flask.palletsprojects.com/en/2.2.x/deploying/waitress/)

To run the application from existing docker image run 
```
docker run -p 5001:5001 pacifikus/dmia_pass_complexity
```
and go to `http://localhost:5001/predict?password={YOUR INPUT}`

To create your own docker image with some modifications run from the project root
```
docker build -t pacifikus/dmia_pass_complexity src/pass_complexity/api
```
