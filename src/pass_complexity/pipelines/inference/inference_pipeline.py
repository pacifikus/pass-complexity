from kedro.pipeline import Pipeline, node
from pass_complexity.pipelines.inference.nodes import predict


def create_pipeline(**kwargs):
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
