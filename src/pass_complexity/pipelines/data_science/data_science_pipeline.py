from kedro.pipeline import Pipeline, node
from pass_complexity.pipelines.data_science.nodes import create_model, split_data


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=split_data,
                inputs=['tokenized_passwords', 'target', "'params:seed"],
                outputs=['X_train', 'X_val', 'y_train', 'y_val'],
                name='train_test_split_node',
                tags=['training'],
            ),
            node(
                func=create_model,
                inputs=['X_train',
                        'X_val',
                        'y_train',
                        'y_val',
                        'params:embedding_vector_length',
                        'params:max_input_length',
                        'params:hidden_dim',
                        'params:epochs'],
                outputs='model',
                name='train_model_node',
                tags=['training'],
            ),
        ],
    )
