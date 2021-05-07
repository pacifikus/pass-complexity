import uvicorn
from fastapi import FastAPI
from kedro.framework.session import KedroSession


app = FastAPI()


@app.get("/{model}/predictor")
async def predictor(model):

    with KedroSession.create("pass_complexity") as session:
        if model == 'lstm_model':
            output = session.run(pipeline_name='api_inference')

    return output


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=6789)
