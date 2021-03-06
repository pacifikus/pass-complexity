import uvicorn
from fastapi import FastAPI
from kedro.framework.session import KedroSession

app = FastAPI()
port = 6789


@app.get('/{model}/predictor')
async def predictor(model):
    """
    Entrypoint to model.

    Args:
        model: model to inference.
    Returns:
         Prediction ouput.
    """
    with KedroSession.create('pass_complexity') as session:
        if model == 'lstm_model':
            output = session.run(pipeline_name='api_inference')

    return output


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=port)
