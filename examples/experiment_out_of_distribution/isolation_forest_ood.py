import os
import sys

current_dir = os.path.dirname(__file__)
repo_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if repo_root not in sys.path:
    sys.path.append(repo_root)

from fastavro import parse_schema
from atelierflow import ExperimentBuilder
from atelierflow.experimentsRunner import ExperimentRunner
from atelierflow.utils.modelConfig import ModelConfig
from atelierflow.metrics.metric import BaseMetric
from mtsa.metrics import calculate_aucroc 
from mtsa.models import IForest  
from examples.experiment_out_of_distribution.isolation_forest_ood_steps import LoadDataStep, PrepareFoldsStep, TrainModelStep, EvaluateModelStep, AppendResultsStep

class ROCAUC(BaseMetric):
    def __init__(self):
        pass

    def compute(self, model, x, y):
        return calculate_aucroc(model, x, y)

def main():
    avro_schema = {
        "namespace": "example.avro",
        "type": "record",
        "name": "ModelResult",
        "fields": [
             {"name": "n_estimators", "type": "string"},
             {"name": "max_features", "type": "string"},
             {"name": "contamination", "type": "string"},
             {"name": "max_samples", "type": "string"},
             {"name": "sampling_rate", "type": "string"},
             {"name": "AUC_ROC", "type": "string"},
         ],
    }

    runner = ExperimentRunner()
    experiment = ExperimentBuilder('Out-of-Distribution Experiment')
    experiment.set_avro_schema(avro_schema)
    
    max_features_values = [0.5, 1.0]
    n_estimators_values = [2, 10, 40, 70, 100]

    iforest_config = ModelConfig(
        model_class=IForest,
        model_parameters={
            "n_estimators": 100,
            "max_samples": 256,
            "contamination": 0.01,
            "max_features": 1.0,
            "sampling_rate": 16000
        },
        model_fit_parameters={},  
    )
    experiment.add_model(iforest_config)

    experiment.add_metric(ROCAUC())

    model_name = iforest_config.model_class.__name__
    folder_name = f"experiment_ood_{model_name}"
    os.makedirs(folder_name, exist_ok=True)
    
    machine_type = "valve"
    output_path = os.path.join(folder_name, f"experiment_ood_results_{machine_type}.avro")

    experiment.add_step(LoadDataStep())
    experiment.add_step(PrepareFoldsStep())
    experiment.add_step(TrainModelStep())
    experiment.add_step(EvaluateModelStep())
    experiment.add_step(AppendResultsStep(output_path, parse_schema(avro_schema)))

    experiments, model_configs, metric_configs = experiment.build()
    runner.add_experiment(experiments, model_configs, metric_configs)

    initial_inputs = {
        "machine_type": machine_type,
        "train_ids": ["id_02", "id_04", "id_06"],
        "test_id": "id_00",
        "max_features_values": max_features_values,
        "n_estimators_values": n_estimators_values
    }

    runner.run_all(initial_input=initial_inputs)

if __name__ == "__main__":
    main()