import os

import pandas as pd
from sklearn.model_selection import train_test_split

from config.config import DATA_PATH, TOKENIZED_DATA, TARGET_DATA, SEED
from src.etl import start_etl
from src.predict import make_prediction
from src.train_pipeline import run_training


def train():
    import datetime

    dif_val = 7
    try:
        creation_date = os.path.getatime(f"{DATA_PATH}{TOKENIZED_DATA}")
        print(f"{DATA_PATH}{TOKENIZED_DATA}")
        now = datetime.datetime.now().timestamp()
        days = (now - creation_date) / 60 / 60 / 24

        if days >= dif_val:
            start_etl()

    except FileNotFoundError:
        start_etl()

    data = pd.read_csv(f"{DATA_PATH}{TOKENIZED_DATA}")
    target = pd.read_csv(f"{DATA_PATH}{TARGET_DATA}")

    X_train, X_val, y_train, y_val = train_test_split(
        data,
        target,
        test_size=0.1,
        random_state=SEED)
    run_training(X_train, y_train, X_val, y_val)


def predict(password):
    make_prediction(password)


if __name__ == "__main__":
    make_prediction('qweqweqwe')
