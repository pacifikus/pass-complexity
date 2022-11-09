"""
This is a boilerplate pipeline 'inference'
generated using Kedro 0.18.3
"""
import re

import pandas as pd
from keras.utils import pad_sequences


def predict_single(model, tokenizer, password, max_input_length):
    """Make prediction for the single string.

    Args:
        model: fitted LSTM model.
        tokenizer: fitted keras tokenizer.
        password: password to strength estimate.
        max_input_length: Max length of input text defined in parameters.yml.
    Returns:
        Strength of the password.

    """
    password = ' '.join(re.findall(r'\S', str(password)))
    token = tokenizer.texts_to_sequences([password])
    token = pad_sequences(token, max_input_length, padding='post')
    predictions = model.predict(token, batch_size=1)
    return predictions[0][0]


def predict(model, test):
    """
    Make prediction for the pandas dataframe.

    Args:
         model: fitted LSTM model.
         test: passwords to strength estimate.
    Returns:
         Strength of the password.

    """
    preds = model.predict(test, batch_size=1024)
    return pd.DataFrame(preds)
