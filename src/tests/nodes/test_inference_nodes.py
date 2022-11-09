import pickle

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies
from hypothesis.extra.pandas import column, data_frames, range_indexes
from keras.models import load_model
from keras.utils import pad_sequences

from pass_complexity.pipelines.inference.nodes import predict, predict_single

max_length = 83


@pytest.fixture(scope='session')
def model():
    return load_model(
        'data/06_models/model.pb',
        compile=False,
    )


@pytest.fixture(scope='session')
def tokenizer():
    with open('data/06_models/tokenizer.pkl', 'rb') as tokenizer_handle:
        return pickle.load(tokenizer_handle)


@given(
    strategies.text(
        min_size=3,
        max_size=max_length,
        alphabet=list('qwertyuiopasdfghjklzxcvbnm'),
    ),
)
@settings(deadline=None)
def test_predict_single(model, tokenizer, test):
    prediction_result = predict_single(model, tokenizer, test, max_length)

    assert isinstance(prediction_result, np.float32)


@given(
    data_frames(
        index=range_indexes(min_size=10, max_size=10),
        columns=[
            column(
                'Password',
                dtype=str,
                elements=strategies.text(
                    min_size=3,
                    max_size=max_length,
                    alphabet=list('abcdef0123456789 '),
                ),
            ),
        ],
    ),
)
@settings(deadline=None)
def test_predict(model, tokenizer, test):
    test_tokens = tokenizer.texts_to_sequences(test['Password'])
    tokenized_test = pad_sequences(test_tokens, max_length, padding='post')
    prediction_result = predict(model, tokenized_test)

    assert isinstance(prediction_result, pd.DataFrame)
    assert len(prediction_result) == len(prediction_result)
