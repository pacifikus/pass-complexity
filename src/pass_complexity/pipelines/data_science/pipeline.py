"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node

from pass_complexity.pipelines.data_science.nodes import create_model, split_data


def create_pipeline(**kwargs):
    """Create data science pipeline.

    Returns:
        Builded pipeline for the ds process: tokenization, data split, model fit.

    """
    return Pipeline(
        [
            node(
                func=split_data,
                inputs=['tokenized_passwords', 'target', "params:training_options"],
                outputs=['X_train', 'X_val', 'y_train', 'y_val'],
                name='train_test_split_node',
                tags=['training'],
            ),
            node(
                func=create_model,
                inputs=[
                    'X_train',
                    'X_val',
                    'y_train',
                    'y_val',
                    'params:training_options',
                ],
                outputs='model',
                name='train_model_node',
                tags=['training'],
            ),
        ],
    )
