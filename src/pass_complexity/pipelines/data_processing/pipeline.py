"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node

from pass_complexity.pipelines.data_processing.nodes import (
    fit_tokenizer,
    preprocess_passwords,
    tokenize_data,
)


def create_pipeline(**kwargs):
    """Create data processing pipeline.

    Returns:
        Builded pipeline for the etl.

    """
    return Pipeline(
        [
            node(
                func=preprocess_passwords,
                inputs=['passwords', 'test'],
                outputs=['preprocessed_passwords', 'target', 'preprocessed_test'],
                name='preprocess_passwords_node',
                tags=['training'],
            ),
            node(
                func=fit_tokenizer,
                inputs='preprocessed_passwords',
                outputs='tokenizer',
                name='prepare_tokenizer_node',
                tags=['training'],
            ),
            node(
                func=tokenize_data,
                inputs=[
                    'tokenizer',
                    'preprocessed_passwords',
                    'preprocessed_test',
                    'params:tokenize_options',
                ],
                outputs=['tokenized_passwords', 'tokenized_test'],
                name='tokenize_passwords_node',
                tags=['training'],
            ),
        ],
    )
