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
from mtsa.models import OSVM  
from examples.experiment_out_of_distribution.isolation_forest_ood_steps import PrepareFoldsStep
from examples.experiment_out_of_distribution.osvm_ood_steps import (
    LoadDataStep, TrainModelStep, EvaluateModelStep, AppendResultsStep
)

class ROCAUC(BaseMetric):
    def __init__(self):
        pass

    def compute(self, model, x, y):
        return calculate_aucroc(model, x, y)

def main():
    avro_schema = {
        "namespace": "example.avro",
        "type": "record",
        "name": "OSVMResult",
        "fields": [
             {"name": "model_name", "type": "string"},
             {"name": "kernel", "type": "string"},
             {"name": "nu", "type": "string"},
             {"name": "sampling_rate", "type": "string"},
             {"name": "train_time_sec", "type": "string"},
             {"name": "AUC_ROC", "type": "string"}
         ],
    }

    runner = ExperimentRunner()
    experiment = ExperimentBuilder('Out-of-Distribution Experiment - OSVM')
    experiment.set_avro_schema(avro_schema)
    
    nu_values = [0.1, 0.05, 0.2]
    kernel_values = ["rbf"]
    sampling_rate_sound = None  
    
    osvm_config = ModelConfig(
        model_class=OSVM,
        model_parameters={
            "nu": 0.1,  
            "kernel": "rbf",
            "sampling_rate": sampling_rate_sound
        },
        model_fit_parameters={},
    )
    experiment.add_model(osvm_config)
    experiment.add_metric(ROCAUC())
 
    model_name = osvm_config.model_class.__name__
    folder_name = f"experiment_ood_{model_name}"
    os.makedirs(folder_name, exist_ok=True)
 
    machine_type = "pump"
    train_ids = ["id_02", "id_04", "id_06"]
    output_path = os.path.join(folder_name, f"experiment_ood_results_{machine_type}_train_ids_{train_ids}.avro")
 
    experiment.add_step(LoadDataStep())
    experiment.add_step(PrepareFoldsStep())
    experiment.add_step(TrainModelStep())
    experiment.add_step(EvaluateModelStep())
    experiment.add_step(AppendResultsStep(output_path, parse_schema(avro_schema)))
 
    experiments, model_configs, metric_configs = experiment.build()
    runner.add_experiment(experiments, model_configs, metric_configs)
 
    initial_inputs = {
        "machine_type": machine_type,
        "train_ids": train_ids,
        "test_id": "id_00",
        "kernel_values": kernel_values,
        "nu_values": nu_values
    }
 
    runner.run_all(initial_input=initial_inputs)
 
if __name__ == "__main__":
    main()