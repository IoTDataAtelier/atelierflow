from fastavro import parse_schema
from atelierflow.experimentsBuilder import ExperimentBuilder
from examples.cross_validation_steps import LoadDataStep, TrainModel, EvaluateModel, CorruptionEvaluationStep, AppendResults

avro_schema = {
    "namespace": "example.avro",
    "type": "record",
    "name": "ModelResult",
    "fields": [
        {"name": "model_name", "type": "string"},
        {"name": "metric_name", "type": "string"},
        {"name": "metric_value", "type": "float"},
        {"name": "date", "type": "string"},
        {"name": "dataset_train", "type": "string"},
        {"name": "dataset_test", "type": "string"},
        {
            "name": "corruption_results",
            "type": {
                "type": "array",
                "items": {
                    "type": "record",
                    "name": "CorruptionResult",
                    "fields": [
                        {"name": "corruption_type", "type": "string"},
                        {"name": "corruption_loss", "type": "float"},
                        {"name": "corruption_accuracy", "type": "float"}
                    ]
                }
            }
        }
    ]
}

output_path = "examples/experiment_results.avro"

builder = ExperimentBuilder()
builder.add_step(LoadDataStep())
builder.add_step(TrainModel())
builder.add_step(EvaluateModel())
builder.add_step(CorruptionEvaluationStep())
builder.add_step(AppendResults(output_path, parse_schema(avro_schema)))


experiments = builder.build()
experiments.run()
