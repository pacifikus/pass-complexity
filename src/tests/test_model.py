import numpy as np
import pytest
from src.main.train_pipeline import create_model
from src.main.predict import make_prediction


@pytest.mark.SMOKE
def test_smoke_model():
    model = create_model()
    assert model is not None, "The model doesn't exist"


@pytest.mark.parametrize("password, expected", [("qwe", 1.8096491), ("qwe123", 1.6993178), ("qwe123!",  1.1572368)])
@pytest.mark.skip
def test_prediction(password, expected):
    assert np.allclose([make_prediction(password)], [expected], atol=1e-06)
