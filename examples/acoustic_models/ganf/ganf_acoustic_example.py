from fastavro import parse_schema
import numpy as np
from atelierflow import BaseModel, ExperimentBuilder
from mtsa.models.ganf import GANF
from ganf_steps import LoadDataStep, PrepareFoldsStep, TrainModelStep, EvaluateModelStep, AppendResultsStep
from atelierflow.metrics.metric import BaseMetric
from mtsa.metrics import calculate_aucroc

class GANFModel(BaseModel):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y, **kwargs):
        self.model.fit(X, y, **kwargs)

    def predict(self, X):
        return self.model.predict(X)

class ROCAUC(BaseMetric):
    def __init__(self):
        pass

    def compute(self, model, y, x):
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
  builder = ExperimentBuilder()
  builder.set_avro_schema(avro_schema)
  sampling_rate_sound = 16000              
  
  initial_inputs = {
    "batch_size_values": np.array([1024, 512, 256, 128, 64, 32]),
    "learning_rate_values": np.array([1e-9]),
    "path_input": "/data/marcelo/pipeflow/examples/sample_data"
  }
  # Instantiate the GANF model
  ganf_model = GANFModel(model=GANF(sampling_rate=sampling_rate_sound, mono=True, use_array2mfcc=True, isForWaveData=True))

  # Add the GANF model to the builder
  builder.add_model(ganf_model)

  # Add the AUC ROC metric
  builder.add_metric(ROCAUC())

  # Define the output path for the Avro file
  output_path = "/data/marcelo/pipeflow/examples/experiment_results.avro"
  builder.add_step(LoadDataStep())
  builder.add_step(PrepareFoldsStep())
  builder.add_step(TrainModelStep())
  builder.add_step(EvaluateModelStep())
  builder.add_step(AppendResultsStep(output_path, parse_schema(avro_schema)))

  # Build the experiments object
  experiments = builder.build()

  # Run the experiments
  experiments.run(initial_inputs)

if __name__ == "__main__":
    main()
