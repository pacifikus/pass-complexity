from keras.callbacks import EarlyStopping
from keras.layers import Dense, LSTM, Embedding, ReLU
from keras.models import Sequential
from keras.optimizers import Adam

from config.config import EMBEDDING_VECTOR_LENGTH, MAX_INPUT_LENGTH, \
    DATA_PATH, MODEL_NAME, HIDDEN_DIM
from src.quality import Losses


def create_model():
    model = Sequential()
    model.add(Embedding(100,
                        EMBEDDING_VECTOR_LENGTH,
                        input_length=MAX_INPUT_LENGTH,
                        mask_zero=True))

    model.add(LSTM(HIDDEN_DIM))
    model.add(Dense(1))
    model.add(ReLU())

    opt = Adam()
    model.compile(loss=Losses.rmsle,
                  optimizer=opt)
    # TODO: log model summary
    return model


def run_training(X_train, y_train, X_val, y_val):
    model = create_model()
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=2,
        verbose=0,
        mode='auto',
    )
    model.fit(
        X_train, y_train,
        epochs=8,
        batch_size=512,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
    )
    # TODO: log history
    model.save(f"{DATA_PATH}{MODEL_NAME}")


if __name__ == '__main__':
    pass
