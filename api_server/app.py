import pandas as pd
import uvicorn
from fastapi import FastAPI


def read_data():
    row = pd.read_csv('data/tokenized_test.csv').sample(1)
    return dict(
        data=row.to_dict(orient='records')[0]
    )


app = FastAPI()


@app.get("/data")
async def index():
    return read_data()


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9876)
