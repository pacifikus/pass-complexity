import string

import pandas as pd
from hypothesis import given, strategies
from hypothesis.extra.pandas import column, data_frames
from keras.preprocessing.text import Tokenizer

from pass_complexity.pipelines.data_processing.nodes import (
    extract_target,
    filter_whitespaces,
    fit_tokenizer,
    preprocess_passwords,
)

max_length = 83
col_pass = 'Password'
col_target = 'Times'
col_id = 'Id'


@given(
    data_frames(
        columns=[
            column(
                col_pass,
                dtype=str,
                elements=strategies.text(
                    min_size=3,
                    max_size=max_length,
                    alphabet=list('abcdef0123456789 '),
                ),
            ),
            column(
                col_target,
                dtype=float,
                elements=strategies.floats(
                    allow_nan=False,
                    allow_infinity=False,
                    min_value=0,
                ),
            ),
        ],
    ),
)
def test_extract_target(data_to_split):
    x_data, target = extract_target(data_to_split)

    assert isinstance(x_data, pd.DataFrame)
    assert isinstance(target, pd.Series)
    assert len(x_data) == len(target)
    assert col_target not in x_data.columns


@given(
    strategies.text(min_size=3, max_size=max_length),
)
def test_filter_whitespaces(input_string):
    res = filter_whitespaces(input_string)
    assert isinstance(res, str)

    res = res.replace(' ', '')
    assert sum([char in res for char in string.whitespace]) == 0


@given(
    data_frames(
        [
            column(
                col_id,
                dtype=int,
                elements=strategies.integers(
                    min_value=0,
                    max_value=1000,
                ),
            ),
            column(
                col_pass,
                dtype=str,
                elements=strategies.text(
                    min_size=3,
                    max_size=max_length,
                    alphabet=list('abcdef0123456789 '),
                ),
            ),
            column(
                col_target,
                dtype=float,
                elements=strategies.floats(
                    allow_nan=False,
                    allow_infinity=False,
                    min_value=0,
                    max_value=10,
                ),
            ),
        ],
    ),
    data_frames(
        [
            column(
                col_id,
                dtype=int,
                elements=strategies.integers(
                    min_value=0,
                    max_value=1000,
                ),
            ),
            column(
                col_pass,
                dtype=str,
                elements=strategies.text(
                    min_size=3,
                    max_size=max_length,
                ),
            ),
        ],
    ),
)
def test_preprocess_passwords(data_to_preprocess, test):
    passwords, target, test = preprocess_passwords(data_to_preprocess, test)

    assert isinstance(passwords, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)
    assert isinstance(target, pd.Series)
    assert len(passwords) == len(target)
    assert 'Id' not in test.columns


@given(
    data_frames(
        [
            column(
                col_id,
                dtype=int,
                elements=strategies.integers(
                    min_value=0,
                    max_value=1000,
                ),
            ),
            column(
                col_pass,
                dtype=str,
                elements=strategies.text(
                    min_size=3,
                    max_size=max_length,
                ),
            ),
            column(
                col_target,
                dtype=float,
                elements=strategies.floats(
                    allow_nan=False,
                    allow_infinity=False,
                    min_value=0,
                    max_value=10,
                ),
            ),
        ],
    ),
)
def test_fit_tokenizer(data_to_fit):
    tokenizer = fit_tokenizer(data_to_fit)

    assert isinstance(tokenizer, Tokenizer)
