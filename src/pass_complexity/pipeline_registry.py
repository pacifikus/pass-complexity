"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline
from pass_complexity.pipelines.data_processing import pipeline as de
from pass_complexity.pipelines.data_science import pipeline as ds
from pass_complexity.pipelines.inference import pipeline as inference
#from pass_complexity.pipelines.api_inference import pipeline as api_inference


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """

    data_processing_pipeline = de.create_pipeline()
    data_science_pipeline = ds.create_pipeline()
    inference_pipeline = inference.create_pipeline()
    #api_inference_pipeline = api_inference.create_pipeline()

    return {
        '__default__':
            data_processing_pipeline +
            data_science_pipeline +
            inference_pipeline,
        'data_processing': data_processing_pipeline,
        'data_science': data_science_pipeline,
        'inference': inference_pipeline,
        #'api_inference': api_inference_pipeline,
    }
