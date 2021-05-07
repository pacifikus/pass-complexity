from kedro.pipeline import Pipeline, node
from pass_complexity.pipelines.api_inference.nodes\
    import get_api_data, predict, serve_result


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=get_api_data,
                inputs="api_data",
                outputs=["predict_date_in", "to_predict_in"],
                tags=["api_inference"],
            ),
            node(
                func=predict,
                inputs=["model", "to_predict_in"],
                outputs="predicted",
                tags=["api_inference"],
            ),
            node(
                func=serve_result,
                inputs=["predict_date_in", "predicted"],
                outputs=["predict_date_out", "predict_out"],
                tags=["api_inference"],
            )
        ]
    )
