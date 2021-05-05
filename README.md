# pass complexity

## Overview

This is a project to predict password complexity generated using `Kedro 0.17.3`.

## How to install dependencies

Declare any dependencies in `src/requirements.txt` for `pip` installation and `src/environment.yml` for `conda` installation.

To install them, run:

```
kedro install
```

## Pipelines

The project includes 3 pipelines:
- data processing pipeline with etl function
- data science pipelines with a train-test split and a model fitting
- inference pipeline with a prediction for the test data


## How to run pipelines

You can run pipelines from the project with:

```
kedro run
```
