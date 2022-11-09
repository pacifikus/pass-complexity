"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.3
"""

import pandas as pd
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense, Embedding, ReLU
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from pass_complexity.pipelines.data_science.quality import Losses
from typing import Dict


def split_data(data_to_split: pd.DataFrame, target: pd.DataFrame, params: Dict):
    """
    Train-test split creation in a 9:1 proportion.

    Args:
        data_to_split: dataset to split.
        target: target to split.
        params: dict with training params, defined in conf/base/parameters/data_science
    Returns:
        Splitted data.
    """
    return train_test_split(
        data_to_split,
        target,
        test_size=0.1,
        random_state=params['seed'],
    )


def create_model(
    x_train: pd.DataFrame,
    x_val: pd.DataFrame,
    y_train: pd.DataFrame,
    y_val: pd.DataFrame,
    params: Dict,
) -> Sequential:
    """
    Tokenize passwords.

    Args:
        x_train: train data.
        x_val: validation data.
        y_train: train target.
        y_val: validation target.
        params: dict with training params, defined in conf/base/parameters/data_science
    Returns:
        Keras Sequential model.
    """
    model = Sequential()
    model.add(
        Embedding(
            100,
            params['embedding_layer_length'],
            input_length=params['max_input_length'],
            mask_zero=True,
        ),
    )

    model.add(LSTM(params['hidden_dim']))
    model.add(Dense(1))
    model.add(ReLU())

    opt = Adam()
    model.compile(
        loss=Losses.rmsle,
        optimizer=opt,
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=2,
        verbose=0,
        mode='auto',
    )
    model.fit(
        x_train,
        y_train,
        epochs=params['epochs'],
        batch_size=params['batch_size'],
        validation_data=(x_val, y_val),
        callbacks=[early_stopping],
    )

    return model
