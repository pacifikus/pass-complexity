from pathlib import Path

import os
import pytest
from src.main.etl import start_etl
from config.config import DATA_PATH, TOKENIZED_DATA, TARGET_DATA


@pytest.fixture(scope='session')
def etl():
    start_etl()


@pytest.mark.skip
@pytest.mark.FILES
def test_train_exists(etl):
    tokenized_train_path = Path(f'{DATA_PATH}{TOKENIZED_DATA}')
    assert tokenized_train_path.exists()
    assert tokenized_train_path.stat().st_size != 0

    os.remove(tokenized_train_path)


@pytest.mark.skip
@pytest.mark.FILES
def test_target_exists(etl):
    y_path = Path(f'{DATA_PATH}{TARGET_DATA}')
    assert y_path.exists()
    assert y_path.stat().st_size != 0
