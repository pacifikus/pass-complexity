"""
This is a boilerplate pipeline 'api_inference'
generated using Kedro 0.18.3
"""
import pandas as pd


def get_api_data(data_to_format):
    """
    Format the data from the api.

    Args:
        data_to_format: data from the request.
    Returns:
         Date from headers, dataframe woth the data.
    """
    return (
        data_to_format.headers['date'],
        pd.DataFrame.from_dict(data_to_format.json()['data'], orient='index').T,
    )


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
    return pd.DataFrame(preds)


def serve_result(predict_date, prediction_results):
    """
    Serve result of the prediction.

    Args:
        predict_date: the date of the prediction.
        prediction_results: predicted value to return.
    Returns:
        Result.
    """
    return predict_date, float(prediction_results[0][0])
