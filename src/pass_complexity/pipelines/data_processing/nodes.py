import re

import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer


def extract_target(data):
    """
    Extract targets from the train.

    Args:
        data: raw train data.
    Returns:
        Train data and targets.
    """
    y = data['Times']
    data.drop(columns='Times', inplace=True)
    return data, y


def filter_whitespaces(x):
    """
    Filter redundant whitespaces from the string.

    Args:
        x: raw string.
    Returns:
        String with redundant whitespaces replaced by a single space.
    """
    x = ' '.join(re.findall(r'\S', str(x)))
    return x


def preprocess_passwords(passwords: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the passwords data.

    Args:
        passwords: raw train data.
        test: raw test data.
    Returns:
        Preprocessed train passwords targets and test.
    """
    passwords, y = extract_target(passwords)
    passwords['Password'] = filter_whitespaces(passwords['Password'])
    test['Password'] = filter_whitespaces(test['Password'])
    test.drop(columns='Id', inplace=True)
    return passwords, y, test


def fit_tokenizer(passwords: pd.DataFrame) -> Tokenizer:
    """
    Fit tokenizer.

    Args:
        passwords: Data for the tokenization.
    Returns:
        Ready-to-work tokenizer.
    """
    tokenizer = Tokenizer(100, filters='', lower=False)
    tokenizer.fit_on_texts(passwords['Password'])
    return tokenizer


def tokenize_data(tokenizer: Tokenizer,
                  passwords: pd.DataFrame,
                  test: pd.DataFrame,
                  max_input_length: int) -> pd.DataFrame:
    """
    Tokenize passwords.

    Args:
        tokenizer: fitted keras tokenizer.
        passwords: passwords for the training.
        max_input_length: Max length of input text defined in parameters.yml.
        test: passwords for the prediction.
    Returns:
        Tokenized train and test passwords
    """
    tokens = tokenizer.texts_to_sequences(passwords['Password'])
    test_tokens = tokenizer.texts_to_sequences(test['Password'])
    tokenized_passwords = pad_sequences(tokens, max_input_length, padding='post')
    tokenized_test = pad_sequences(test_tokens, max_input_length, padding='post')
    return pd.DataFrame(tokenized_passwords), pd.DataFrame(tokenized_test)
