import numpy as np
from atelierflow import BaseModel, Experiments, BaseMetric, Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

class SKLearnModel(BaseModel):
  def __init__(self, model, fit_params=None, predict_params=None):
    self.model = model
    self.fit_params = fit_params or {}
    self.predict_params = predict_params or {}

  def fit(self, X, y, **kwargs):
    self.model.fit(X, y, **kwargs)

  def predict(self, X, **kwargs):
    return self.model.predict(X, **kwargs)

  def get_parameters_description(self):
    return {
        "learning_rate": "1e-10", 
        "epoch": "100",
        "model_version": "1.0"
    }

  def get_fit_params(self):
    return self.fit_params

  def get_predict_params(self):
    return self.predict_params

  def requires_supervised_data(self):
    return True

class AccuracyMetric(BaseMetric):
  def __init__(self, name=None, compute_params=None):
    super().__init__(name, compute_params)

  def compute(self, y_true, y_pred):
    return accuracy_score(y_true, y_pred)
  
  def get_compute_params(self):
    return super().get_compute_params()

class F1Metric(BaseMetric):
  def __init__(self, name=None, compute_params=None):
    super().__init__(name, compute_params)

  def compute(self, y_true, y_pred):
    return f1_score(y_true, y_pred, average="weighted")
  
  def get_compute_params(self):
    return super().get_compute_params()

def main():
  # Define the Avro schema for saving results
  avro_schema = {
      "namespace": "example.avro",
      "type": "record",
      "name": "ModelResult",
      "fields": [
          {"name": "model_name", "type": "string"},
          {"name": "metric_name", "type": "string"},
          {"name": "metric_value", "type": "float"},
          {"name": "model_version", "type": "string", "default": "null"},
          {"name": "date", "type": "string"},
          {"name": "dataset_train", "type": "string"},
          {"name": "dataset_test", "type": "string"},
          {"name": "learning_rate", "type": "string"},
          {"name": "epoch", "type": "string"},
      ],
  }

  # Create experiments with cross-validation
  exp = Experiments(avro_schema=avro_schema, cross_validation=True, n_splits=5)

  # Add model and metrics to the experiment
  exp.add_model(SKLearnModel(LogisticRegression()))
  exp.add_metric(AccuracyMetric(name="accuracy"))
  exp.add_metric(F1Metric(name='f1'))

  # Create the datasets
  train_1 = Dataset("train1", X_train=np.random.rand(100, 5), y_train=np.random.randint(0, 2, 100))
  test_2 = Dataset("test2", X_test=np.random.rand(75, 5), y_test=np.random.randint(0, 2, 75))

  # Add datasets to the experiment
  exp.add_train(train_1)
  exp.add_test(test_2)

  # Run experiments and save results to Avro
  exp.run("examples/experiment_results.avro")

  # To view the experiment results, run this code in your terminal: "python pipeflow/read_avro"

if __name__ == "__main__":
  main()
