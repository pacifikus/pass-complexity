import pickle

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, strategies, settings
from hypothesis.extra.pandas import data_frames, column, range_indexes
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

from pass_complexity.pipelines.inference.nodes import predict, predict_single


@pytest.fixture(scope='session')
def model():
    return load_model(
        f'data/06_models/model.pb',
        compile=False
    )


@pytest.fixture(scope='session')
def tokenizer():
    with open('data/06_models/tokenizer.pkl', 'rb') as handle:
        return pickle.load(handle)


@given(
    strategies.text(min_size=3,
                    max_size=83,
                    alphabet=list('qwertyuiopasdfghjklzxcvbnm'))
)
@settings(deadline=None)
def test_predict_single(model, tokenizer, test):
    result = predict_single(model, tokenizer, test, 83)

    assert isinstance(result, np.float32)


@given(
    data_frames(
        index=range_indexes(min_size=10, max_size=10),
        columns=[
            column('Password', dtype=str,
                   elements=strategies.text(min_size=3,
                                            max_size=83,
                                            alphabet=list('abcdef0123456789 '))),
        ]
    )
)
@settings(deadline=None)
def test_predict(model, tokenizer, test):
    test_tokens = tokenizer.texts_to_sequences(test['Password'])
    tokenized_test = pad_sequences(test_tokens, 83, padding='post')
    result = predict(model, tokenized_test)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(result)