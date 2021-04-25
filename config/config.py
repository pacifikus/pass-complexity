import os


ROOT = r'%s' % os.path.abspath(
    os.path.join(os.path.dirname("src"))
).replace('\\', '/')

DATA_PATH = r'%s' % os.path.abspath(
    os.path.join(os.path.dirname("src"), 'data')
).replace('\\', '/')

TRAIN_DATA = "/raw_data/train.csv"
TOKENIZED_DATA = "/cleaned_data/tokenized.csv"
TARGET_DATA = "/cleaned_data/target.csv"
TOKENIZER = "/cleaned_data/tokenizer.pickle"
MODEL_NAME = "/model"

SEED = 42
EMBEDDING_VECTOR_LENGTH = 16
HIDDEN_DIM = 16
MAX_INPUT_LENGTH = 83
