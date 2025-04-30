import os
import sys


module_path = os.path.abspath(os.path.join('../pipeflow/'))
if module_path not in sys.path:
    sys.path.append(module_path)

from fastavro import parse_schema
import numpy as np
from atelierflow import ExperimentBuilder
from mtsa.models.ganf import GANF
from mtsa.models.ransyncorders import RANSynCoders
from mtsa.models.isolationforest import IForest
from mtsa.models.oneClassSVM import OneClassSVM
from mtsa.models.hitachi import Hitachi
from steps import LoadDataStep, PrepareFoldsStep, TrainModelStep, EvaluateModelStep, AppendResultsStep
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
      {
        "name": "model_name",
        "type": "string"
      },
      {
        "name": "metrics",
        "type": {
          "type": "map",
          "values": "string"  
        }
      },
      {
        "name": "additional",
        "type": {
          "type": "map",
          "values": "string" 
        }
      }
    ]
  }

  output_path = "/data/marcelo/pipeflow/examples/experiment_henrique.avro"           


  runner = ExperimentRunner()
  experiment1 = ExperimentBuilder('Mtsa Experiments')
  experiment1.set_avro_schema(avro_schema)

  isolation = ModelConfig(
    model_class=IForest,
  )
  experiment1.add_model(isolation)
  
  hitachi = ModelConfig(
    model_class=Hitachi,
  )
  experiment1.add_model(hitachi)

  oneClass = ModelConfig(
    model_class=OneClassSVM,
    model_parameters={"kernel":"rbf", "nu":0.1}
  )
  experiment1.add_model(oneClass)

  experiment1.add_metric(ROCAUC)

  experiment1.add_step(LoadDataStep())
  experiment1.add_step(PrepareFoldsStep())
  experiment1.add_step(TrainModelStep())
  experiment1.add_step(EvaluateModelStep())
  experiment1.add_step(AppendResultsStep(output_path, parse_schema(avro_schema)))

  experiments, model_configs, metric_configs = experiment1.build()
  runner.add_experiment(experiments, model_configs, metric_configs) 


  initial_inputs = [
    os.path.join(os.getcwd(), "..", "..", "henrique", "EmbeddedAI","lacci2025","in_distribution","A29v","config1config2"),
    os.path.join(os.getcwd(), "..", "..", "henrique", "EmbeddedAI","lacci2025","in_distribution","A29v","config1config3"),
    os.path.join(os.getcwd(), "..", "..", "henrique", "EmbeddedAI","lacci2025","in_distribution","A29v","config1config4"),
    os.path.join(os.getcwd(), "..", "..", "henrique", "EmbeddedAI","lacci2025","in_distribution","A29v","config1config5"),
  ]


  runner.run_all(initial_input=initial_inputs)

if __name__ == "__main__":
  main()
