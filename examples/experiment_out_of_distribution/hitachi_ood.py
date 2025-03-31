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
from mtsa.models import Hitachi  
from examples.experiment_out_of_distribution.isolation_forest_ood_steps import PrepareFoldsStep
from examples.experiment_out_of_distribution.ganf_ood_steps import LoadDataStep, EvaluateModelStep
from examples.experiment_out_of_distribution.hitachi_ood_steps import TrainModelStep, AppendResultsStep

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
             {"name": "model_name", "type": "string"},
             {"name": "batch_size", "type": "string"},
             {"name": "epoch_size", "type": "string"},
             {"name": "learning_rate", "type": "string"},
             {"name": "sampling_rate", "type": "string"},
             {"name": "train_time_sec", "type": "string"},
             {"name": "AUC_ROCs", "type": "string"},
         ],
     }
     
    runner = ExperimentRunner()
    experiment = ExperimentBuilder('Out-of-Distribution Hitachi Experiment')
    experiment.set_avro_schema(avro_schema)
    
    batch_size_values = [512, 360, 128, 90, 32]
    learning_rate_values = [1e-3, 1e-6]
    sampling_rate_sound = 16000

    hitachi_config = ModelConfig(
        model_class=Hitachi,
        model_parameters={
            "sampling_rate": sampling_rate_sound,
            "n_mels": 64,
            "frames": 5,
            "n_fft": 1024,
            "hop_length": 512,
            "power": 2.0,
            "mono": False,
            "epochs": 50,            
            "batch_size": 512,       
            "learning_rate": 1e-3,     
            "shuffle": True,
            "validation_split": 0.1,
            "verbose": 0
        },
        model_fit_parameters={} 
    )
    experiment.add_model(hitachi_config)
    
    experiment.add_metric(ROCAUC())
    
    model_name = hitachi_config.model_class.__name__
    folder_name = f"experiment_ood_{model_name}"
    os.makedirs(folder_name, exist_ok=True)
    
    machine_type = "slider"
    output_path = os.path.join(folder_name, f"experiment_ood_results_{machine_type}.avro")
    
    experiment.add_step(LoadDataStep())
    experiment.add_step(PrepareFoldsStep())
    experiment.add_step(TrainModelStep())
    experiment.add_step(EvaluateModelStep())
    experiment.add_step(AppendResultsStep(output_path, parse_schema(avro_schema)))
    
    experiments, model_configs, metric_configs = experiment.build()
    runner.add_experiment(experiments, model_configs, metric_configs)
    
    initial_inputs = {
         "machine_type":machine_type,  
         "train_ids": ["id_02", "id_04", "id_06"],
         "test_id": "id_00",
         "batch_size_values": batch_size_values,
         "learning_rate_values": learning_rate_values
    }
    
    runner.run_all(initial_input=initial_inputs)

if __name__ == "__main__":
    main()