"""
This is a boilerplate pipeline 'inference'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node

from pass_complexity.pipelines.inference.nodes import predict


def create_pipeline(**kwargs):
    """Create inference pipeline.

    Returns:
        Builded pipeline for the inference.

    """
    return Pipeline(
        [
            node(
                func=predict,
                inputs=['model', 'tokenized_test'],
                outputs='predictions',
                name='inference_node',
                tags=['inference'],
            ),
        ],
    )
