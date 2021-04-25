from keras.models import load_model

from config.config import DATA_PATH, MODEL_NAME
from src.preprocessors import CustomTokenizer
from src.quality import Losses


def make_prediction(password):
    model = load_model(
        f'{DATA_PATH}{MODEL_NAME}',
        custom_objects={'rmsle': Losses.rmsle},
    )
    token = CustomTokenizer().tokenize_single_pass(password)
    predictions = model.predict(token, batch_size=1)
    return predictions[0][0]


if __name__ == '__main__':
    make_prediction('qweqweqwe')
