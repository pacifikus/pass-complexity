import logging
from typing import Dict, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

from keras.callbacks import EarlyStopping
from keras.layers import Dense, LSTM, Embedding, ReLU
from keras.models import Sequential
from keras.optimizers import Adam

from pass_complexity.pipelines.data_science.quality import Losses


def split_data(data: pd.DataFrame, target: pd.DataFrame, seed: int):
    """Train-test split creation in a 9:1 proportion.

    Args:
        data: dataset to split.
        target: target to split.
        seed: random state defined in parameters.yml.
    Returns:
        Splitted data.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        data,
        target,
        test_size=0.1,
        random_state=seed)
    return X_train, X_val, y_train, y_val


def create_model(X_train: pd.DataFrame,
                 X_val: pd.DataFrame,
                 y_train: pd.DataFrame,
                 y_val: pd.DataFrame,
                 embedding_layer_length: int,
                 max_input_length: int,
                 hidden_dim: int,
                 epochs: int,) -> Sequential:
    """Tokenize passwords

    Args:
        X_train: train data.
        X_val: validation data.
        y_train: train target.
        y_val: validation target.
        embedding_layer_length: Embedding layer output_dim defined in parameters.yml.
        max_input_length: Max length of input text defined in parameters.yml.
        hidden_dim: dimensionality of the LSTM output space defined in parameters.yml.
        epochs: number of training epochs defined in parameters.yml.
    Returns:
        Keras Sequential model.
    """
    model = Sequential()
    model.add(Embedding(100,
                        embedding_layer_length,
                        input_length=max_input_length,
                        mask_zero=True))

    model.add(LSTM(hidden_dim))
    model.add(Dense(1))
    model.add(ReLU())

    opt = Adam()
    model.compile(loss=Losses.rmsle,
                  optimizer=opt)

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=2,
        verbose=0,
        mode='auto',
    )
    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=512,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
    )

    return model
