import string

import pandas as pd
from hypothesis import given, strategies
from hypothesis.extra.pandas import data_frames, column
from keras.preprocessing.text import Tokenizer

from pass_complexity.pipelines.data_processing.nodes \
    import extract_target, filter_whitespaces, preprocess_passwords, fit_tokenizer


@given(
    data_frames(
        columns=[
            column('Password', dtype=str,
                   elements=strategies.text(min_size=3,
                                            max_size=83,
                                            alphabet=list('abcdef0123456789 '))),
            column('Times', dtype=float,
                   elements=strategies.floats(allow_nan=False,
                                              allow_infinity=False,
                                              min_value=0)),
        ]
    )
)
def test_extract_target(data):
    x, y = extract_target(data)

    assert isinstance(x, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(x) == len(y)
    assert 'Times' not in x.columns


@given(
    strategies.text(min_size=3, max_size=83)
)
def test_filter_whitespaces(input_string):
    res = filter_whitespaces(input_string)
    assert isinstance(res, str)

    res = res.replace(' ', '')
    assert sum([c in res for c in string.whitespace]) == 0


@given(
    data_frames(
        [
            column('Id', dtype=int,
                   elements=strategies.integers(min_value=0,
                                                max_value=1000)),
            column('Password', dtype=str,
                   elements=strategies.text(min_size=3,
                                            max_size=83,
                                            alphabet=list('abcdef0123456789 '))),
            column('Times', dtype=float,
                   elements=strategies.floats(allow_nan=False,
                                              allow_infinity=False,
                                              min_value=0, max_value=10)),
        ]
    ),
    data_frames(
        [
            column('Id', dtype=int,
                   elements=strategies.integers(min_value=0,
                                                max_value=1000)),
            column('Password', dtype=str,
                   elements=strategies.text(min_size=3,
                                            max_size=83)),
        ]
    )
)
def test_preprocess_passwords(data, test):
    passwords, y, test = preprocess_passwords(data, test)

    assert isinstance(passwords, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(passwords) == len(y)
    assert 'Id' not in test.columns


@given(
    data_frames(
        [
            column('Id', dtype=int,
                   elements=strategies.integers(min_value=0,
                                                max_value=1000)),
            column('Password', dtype=str,
                   elements=strategies.text(min_size=3,
                                            max_size=83)),
            column('Times', dtype=float,
                   elements=strategies.floats(allow_nan=False,
                                              allow_infinity=False,
                                              min_value=0,
                                              max_value=10)),
        ]
    )
)
def test_fit_tokenizer(data):
    tokenizer = fit_tokenizer(data)

    assert isinstance(tokenizer, Tokenizer)
