import pandas as pd
from hypothesis import given, strategies
from hypothesis.extra.pandas import data_frames, column, range_indexes

from pass_complexity.pipelines.data_science.nodes import split_data


@given(
    data_frames(
        index=range_indexes(min_size=10, max_size=10),
        columns=[
            column('Password', dtype=str,
                   elements=strategies.text(min_size=3,
                                            max_size=83,
                                            alphabet=list('abcdef0123456789 '))),
        ]
    ),
    data_frames(
        index=range_indexes(min_size=10, max_size=10),
        columns=[
            column('Times', dtype=float,
                   elements=strategies.floats(allow_nan=False,
                                              allow_infinity=False,
                                              min_value=0))
        ]
    ),
    strategies.integers(min_value=0, max_value=1000)
)
def test_split_data(X, y, seed):
    X_train, X_val, y_train, y_val = split_data(X, y, seed)

    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_val, pd.DataFrame)
    assert isinstance(y_train, pd.DataFrame)
    assert isinstance(y_val, pd.DataFrame)