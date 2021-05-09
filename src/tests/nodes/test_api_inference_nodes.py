"""Tests for api_inference nodes."""

import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies
from hypothesis.extra.numpy import arrays as np_arrays
from hypothesis.extra.pandas import column, data_frames, range_indexes
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from pass_complexity.pipelines.api_inference.nodes import predict, serve_result


@pytest.fixture(scope='session')
def model():
    return load_model(
        'data/06_models/model.pb',
        compile=False,
    )


@pytest.fixture(scope='session')
def tokenizer():
    with open('data/06_models/tokenizer.pkl', 'rb') as handle:
        return pickle.load(handle)


@given(
    data_frames(
        index=range_indexes(min_size=10, max_size=10),
        columns=[
            column('Password', dtype=str,
                   elements=strategies.text(min_size=3,
                                            max_size=83,
                                            alphabet=list('abcdef0123456789 '))),
        ],
    ),
)
@settings(deadline=None)
def test_predict(model, tokenizer, test):
    test_tokens = tokenizer.texts_to_sequences(test['Password'])
    tokenized_test = pad_sequences(test_tokens, 83, padding='post')
    result = predict(model, tokenized_test)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(result)


@given(
    np_arrays(dtype=np.float32, shape=(1, 1, 1)),
)
def test_serve_result(predict):
    result_date, result_value = serve_result(datetime.now(), predict)

    assert isinstance(result_value, float)
    assert isinstance(result_date, datetime)
