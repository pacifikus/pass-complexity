import re

import pandas as pd

from config.config import DATA_PATH, TRAIN_DATA, TOKENIZED_DATA, TARGET_DATA
from src.main.preprocessors import CustomTokenizer


def start_etl():
    train = pd.read_csv(f'{DATA_PATH}{TRAIN_DATA}')
    y = train.Times

    train.drop(columns='Times', inplace=True)
    train['Password'] = train.Password.apply(
        lambda x: ' '.join(re.findall(r'\S', str(x)))
    )

    tokenizer = CustomTokenizer()
    tokenized_data = pd.DataFrame(tokenizer.tokenize_df(train))

    tokenized_data.to_csv(
        f"{DATA_PATH}{TOKENIZED_DATA}",
        header=True,
        index=False,
    )
    y.to_csv(f"{DATA_PATH}{TARGET_DATA}", header=True, index=False)


if __name__ == "__main__":
    pass
