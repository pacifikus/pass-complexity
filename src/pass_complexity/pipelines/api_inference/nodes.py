import pandas as pd


def get_api_data(data):
    return (data.headers['date'],
            pd.DataFrame.from_dict(data.json()['data'], orient='index').T)


def predict(model, test):
    """
    Predict for the pandas dataframe.

    Args:
        model: fitted LSTM model.
        test: passwords to strength estimate.
    Returns:
         Strength of the password.
    """
    preds = model.predict(test, batch_size=1024)
    result = pd.DataFrame(preds)
    return result


def serve_result(predict_date, predict):
    """
    Serve result of the prediction.

    Args:
        predict_date: the date of the prediction.
        predict: predicted value to return.
    Returns:
        Result.
    """
    return predict_date, float(predict[0][0])
