import re
import pandas as pd
from keras.preprocessing.sequence import pad_sequences


def predict_single(model, tokenizer, password, max_input_length):
    """
    Args:
        model: fitted LSTM model.
        tokenizer: fitted keras tokenizer.
        password: password to strength estimate.
        max_input_length: Max length of input text defined in parameters.yml.
    Returns:
        Strength of the password.
    """
    password = " ".join(re.findall(r"\S", str(password)))
    token = tokenizer.texts_to_sequences([password])
    token = pad_sequences(token, max_input_length, padding="post")
    predictions = model.predict(token, batch_size=1)
    return predictions[0][0]


def predict(model, test):
    preds = model.predict(test, batch_size=1024)
    result = pd.DataFrame(preds)
    return result