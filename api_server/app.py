"""Api server.

This module run app to return one string from the test data.
The app runs on :9876 port.
"""

import pandas as pd
import uvicorn
from fastapi import FastAPI


def read_data():
    """
    Return a single string from the test dataset.

    Returns:
         One password to inference.
    """
    row = pd.read_csv('data/tokenized_test.csv').sample(1)
    return {
        'data': row.to_dict(orient='records')[0],
    }


app = FastAPI()


@app.get('/data')
async def index():
    """
    Entrypoint to api server.

    Returns:
         Read data function.
    """
    return read_data()


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=9876)
