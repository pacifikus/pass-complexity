# Pipeline data_science

## Overview

The pipeline for training LSTM regression model via Keras framework 

## Pipeline inputs

- Tokenized train data of the `pandas.CSVDataSet` datatype from `data/03_primary` 
- Target - `pandas.CSVDataSet` datatype from `data/03_primary`
- training_options from `conf/base/data_science.yml`

## Pipeline outputs

Trained LSTM model and tokenizer in the `data/06_models`
