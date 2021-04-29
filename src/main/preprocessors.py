import pickle
import re

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from config.config import MAX_INPUT_LENGTH, DATA_PATH, TOKENIZER


class CustomTokenizer:
    def __init__(self):
        try:
            self._load_tokenizer()
            self.is_fitted = True
        except FileNotFoundError:
            self.tokenizer = Tokenizer(100, filters='', lower=False)
            self.is_fitted = False

    def tokenize_df(self, df):
        if not self.is_fitted:
            self.tokenizer.fit_on_texts(df['Password'])
            self._save_tokenizer()
        tokens = self.tokenizer.texts_to_sequences(df['Password'])
        df_tokenized = pad_sequences(tokens, MAX_INPUT_LENGTH, padding='post')
        return df_tokenized

    def tokenize_single_pass(self, password):
        password = " ".join(re.findall(r"\S", str(password)))
        tokens = self.tokenizer.texts_to_sequences([password])
        tokens = pad_sequences(tokens, MAX_INPUT_LENGTH, padding="post")
        return tokens

    def _save_tokenizer(self):
        with open(f"{DATA_PATH}{TOKENIZER}", 'wb') as handle:
            pickle.dump(
                self.tokenizer,
                handle,
                protocol=pickle.HIGHEST_PROTOCOL
            )

    def _load_tokenizer(self):
        with open(f"{DATA_PATH}{TOKENIZER}", 'rb') as handle:
            self.tokenizer = pickle.load(handle)
