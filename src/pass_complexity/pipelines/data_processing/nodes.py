import re

import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

col_pass = 'Password'
col_target = 'Times'
col_id = 'Id'


def extract_target(data_to_process):
    """
    Extract targets from the train.

    Args:
        data_to_process: raw train data.
    Returns:
        Train data and targets.
    """
    target = data_to_process[col_target]
    data_to_process.drop(columns=col_target, inplace=True)
    return data_to_process, target


def filter_whitespaces(input_str):
    """
    Filter redundant whitespaces from the string.

    Args:
        input_str: raw string.
    Returns:
        String with redundant whitespaces replaced by a single space.
    """
    return ' '.join(re.findall(r'\S', str(input_str)))


def preprocess_passwords(passwords: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the passwords data.

    Args:
        passwords: raw train data.
        test: raw test data.
    Returns:
        Preprocessed train passwords targets and test.
    """
    passwords, target = extract_target(passwords)
    passwords[col_pass] = filter_whitespaces(passwords[col_pass])
    test[col_pass] = filter_whitespaces(test[col_pass])
    test.drop(columns=col_id, inplace=True)
    return passwords, target, test


def fit_tokenizer(passwords: pd.DataFrame) -> Tokenizer:
    """
    Fit tokenizer.

    Args:
        passwords: Data for the tokenization.
    Returns:
        Ready-to-work tokenizer.
    """
    tokenizer = Tokenizer(100, filters='', lower=False)
    tokenizer.fit_on_texts(passwords[col_pass])
    return tokenizer


def tokenize_data(
    tokenizer: Tokenizer,
    passwords: pd.DataFrame,
    test: pd.DataFrame,
    max_input_length: int,
) -> pd.DataFrame:
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
    tokens = tokenizer.texts_to_sequences(passwords[col_pass])
    test_tokens = tokenizer.texts_to_sequences(test[col_pass])
    tokenized_passwords = pad_sequences(tokens, max_input_length, padding='post')
    tokenized_test = pad_sequences(test_tokens, max_input_length, padding='post')
    return pd.DataFrame(tokenized_passwords), pd.DataFrame(tokenized_test)
