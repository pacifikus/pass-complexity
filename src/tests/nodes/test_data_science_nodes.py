import pandas as pd
from hypothesis import given, strategies
from hypothesis.extra.pandas import column, data_frames, range_indexes

from pass_complexity.pipelines.data_science.nodes import split_data

max_length = 83


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
    data_frames(
        index=range_indexes(min_size=10, max_size=10),
        columns=[
            column(
                'Times',
                dtype=float,
                elements=strategies.floats(
                    allow_nan=False,
                    allow_infinity=False,
                    min_value=0,
                ),
            ),
        ],
    ),
    strategies.integers(min_value=0, max_value=1000),
)
def test_split_data(x_data, y_data, seed):
    x_train, x_val, y_train, y_val = split_data(x_data, y_data, seed)

    assert isinstance(x_train, pd.DataFrame)
    assert isinstance(x_val, pd.DataFrame)
    assert isinstance(y_train, pd.DataFrame)
    assert isinstance(y_val, pd.DataFrame)
