# pass complexity

[![Tests](https://github.com/pacifikus/pass-complexity/actions/workflows/tests.yml/badge.svg)](https://github.com/pacifikus/pass-complexity/actions/workflows/tests.yml)
[![Code review](https://github.com/pacifikus/pass-complexity/actions/workflows/code-review.yml/badge.svg)](https://github.com/pacifikus/pass-complexity/actions/workflows/code-review.yml)
[![wemake-python-styleguide](https://img.shields.io/badge/style-wemake-000000.svg)](https://github.com/wemake-services/wemake-python-styleguide)


## Overview

This is a project to predict password complexity

## Install dependencies

First of all, install project dependencies, with command:

```
pip install -r src/requirements.txt
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
