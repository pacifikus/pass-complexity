# pass complexity

[![Tests](https://github.com/pacifikus/pass-complexity/actions/workflows/tests.yml/badge.svg)](https://github.com/pacifikus/pass-complexity/actions/workflows/tests.yml)

## Overview

This is a project to predict password complexity

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

Current pipeline DAG looks like:

![Pipeline DAG](https://github.com/pacifikus/pass-complexity/blob/main/imgs/viz_pipelines.png)


## How to run pipelines

You can run pipelines from the project with:

```
kedro run
```
