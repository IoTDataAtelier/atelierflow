from fastavro import parse_schema
import numpy as np
from atelierflow import ExperimentBuilder
from atelierflow.model import BaseModel
from mtsa.models.ganf import GANF
from mtsa.models.hitachi import Hitachi
from ganf_steps import LoadDataStep, PrepareFoldsStep, TrainModelStep, EvaluateModelStep, AppendResultsStep
from atelierflow.metrics.metric import BaseMetric
from atelierflow.experimentsRunner import ExperimentRunner
from mtsa.metrics import calculate_aucroc
from atelierflow.utils.modelConfig import ModelConfig

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
      {"name": "AUC_ROCs", "type": "string"},
    ],
  }
  output_path = "/data/marcelo/pipeflow/examples/experiment_results.avro"           


  runner = ExperimentRunner()
  experiment1 = ExperimentBuilder('Mtsa Experiments')
  experiment1.set_avro_schema(avro_schema)

 
  hitachi_config = ModelConfig(
    model_class=Hitachi,
    model_parameters={},
    model_fit_parameters={},
  )
  experiment1.add_model(hitachi_config)

  ganf_config = ModelConfig(
    model_class=GANF,
    model_parameters={"sampling_rate": 16000, "mono": True, "use_array2mfcc": True, "isForWaveData": True},
    model_fit_parameters={"batch_size": 32, "learning_rate": 1e-4, "epochs": 2},
  )
  experiment1.add_model(ganf_config)
  

  experiment1.add_metric(ROCAUC)

  experiment1.add_step(LoadDataStep())
  experiment1.add_step(PrepareFoldsStep())
  experiment1.add_step(TrainModelStep())
  experiment1.add_step(EvaluateModelStep())
  experiment1.add_step(AppendResultsStep(output_path, parse_schema(avro_schema)))

  experiments, model_configs, metric_configs = experiment1.build()
  runner.add_experiment(experiments, model_configs, metric_configs) 


  initial_inputs = {
    "path": "/data/marcelo/pipeflow/examples/sample_data/machine_type_1/id_00",
  }

  runner.run_all(initial_input=initial_inputs)

if __name__ == "__main__":
    main()
