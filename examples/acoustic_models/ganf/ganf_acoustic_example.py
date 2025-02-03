from fastavro import parse_schema
import numpy as np
from atelierflow import ExperimentBuilder
from mtsa.models.ganf import GANF
from ganf_steps import LoadDataStep, PrepareFoldsStep, TrainModelStep, EvaluateModelStep, AppendResultsStep
from atelierflow.metrics.metric import BaseMetric
from mtsa.metrics import calculate_aucroc

class ROCAUC(BaseMetric):
    def __init__(self):
        pass

    def compute(self, model, x, y):
        return calculate_aucroc(model, x, y)

def main():

  # Define the Avro schema for saving results
  avro_schema = {
      "namespace": "example.avro",
      "type": "record",
      "name": "ModelResult",
      "fields": [
          {"name": "batch_size", "type": "string"},
          {"name": "epoch_size", "type": "string"},
          {"name": "learning_rate", "type": "string"},
          {"name": "sampling_rate", "type": "string"},
          {"name": "AUC_ROCs", "type": "string"},
      ],
  }

  # Instantiate the ExperimentBuilder
  experiment1 = ExperimentBuilder()
  experiment1.set_avro_schema(avro_schema)
  sampling_rate_sound = 16000              
  
  # Add the GANF model to the builder
  experiment1.add_model(GANF, sampling_rate=sampling_rate_sound, mono=True, use_array2mfcc=True, isForWaveData=True)

  # Add the AUC ROC metric
  experiment1.add_metric(ROCAUC())

  # Define the output path for the Avro file
  output_path = "/data/marcelo/pipeflow/examples/experiment_results.avro"
  experiment1.add_step(LoadDataStep())
  experiment1.add_step(PrepareFoldsStep())
  experiment1.add_step(TrainModelStep())
  experiment1.add_step(EvaluateModelStep())
  experiment1.add_step(AppendResultsStep(output_path, parse_schema(avro_schema)))

  # Build the experiments object
  experiments, model_kwargs = experiment1.build()

  initial_inputs = {
    "batch_size_values": np.array([32]),
    "learning_rate_values": np.array([1e-9]),
    "path": "/data/MIMII/slider/id_00",
    'model_kwargs': model_kwargs
  }

  # Run the experiments
  experiments.run(initial_inputs)

if __name__ == "__main__":
    main()
