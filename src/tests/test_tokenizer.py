from pathlib import Path

import pytest
from src.main.preprocessors import CustomTokenizer
from config.config import DATA_PATH, TOKENIZER


@pytest.mark.FILES
def test_is_tokenizer_file_exists():
    tokenizer_path = Path(f'{DATA_PATH}{TOKENIZER}')
    assert tokenizer_path.exists()
    assert CustomTokenizer() is not None


@pytest.mark.SMOKE
def test_smoke_tokenizer():
    assert CustomTokenizer() is not None

