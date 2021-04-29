import pytest
import numpy as np
from sklearn.metrics import mean_squared_log_error
from src.main.quality import Losses


@pytest.mark.parametrize("y_true, y_pred", [([10, 100], [1, 2]), ([1, 2, 3], [1, 2, 3]), ([0, 0, 0], [0, 0, 0])])
def test_rmsle(y_true, y_pred):
    sklearn_rmsle = np.sqrt(mean_squared_log_error(y_true, y_pred))
    assert Losses.rmsle(y_true, y_pred) == sklearn_rmsle
