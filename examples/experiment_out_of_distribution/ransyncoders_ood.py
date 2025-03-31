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
from mtsa.models import RANSynCoders  
from examples.experiment_out_of_distribution.isolation_forest_ood_steps import PrepareFoldsStep
from examples.experiment_out_of_distribution.ganf_ood_steps import LoadDataStep, EvaluateModelStep, AppendResultsStep
from examples.experiment_out_of_distribution.ransyncoders_ood_steps import TrainModelStep
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
             {"name": "AUC_ROCs", "type": "string"}
         ],
    }

    runner = ExperimentRunner()
    experiment = ExperimentBuilder('Out-of-Distribution RANSynCoders Experiment')
    experiment.set_avro_schema(avro_schema)

    batch_size_values = ([720, 360, 180, 90, 45])
    learning_rate_values = ([1e-3, 1e-6])
    sampling_rate_sound = 16000 

    ransyncoders_config = ModelConfig(
        model_class=RANSynCoders,
        model_parameters={
            "is_acoustic_data": True,
            "mono": True,
            "normal_classifier": 1,
            "abnormal_classifier": 0,
            "synchronize": True,
            "sampling_rate": sampling_rate_sound,
            "batch_size": 720,         
            "learning_rate": 1e-3,      
            "epochs": 20               
        },
        model_fit_parameters={}
    )
    experiment.add_model(ransyncoders_config)

    experiment.add_metric(ROCAUC())

    model_name = ransyncoders_config.model_class.__name__
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
         "batch_size_values": batch_size_values,
         "learning_rate_values": learning_rate_values
    }

    runner.run_all(initial_input=initial_inputs)

if __name__ == "__main__":
    main()