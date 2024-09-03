import numpy as np

from atelierflow import BaseModel, Experiments, BaseMetric, Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

class SKLearnModel(BaseModel):
  def __init__(self, model):
    self.model = model

  def fit(self, X, y):
    self.model.fit(X, y)

  def predict(self, X):
    return self.model.predict(X)  
  
  def get_parameters_description(self):
    return {
      "learning_rate": "1e-10", 
      "epoch": "100",
      "model_version": "1.0"
    }

class AccuracyMetric(BaseMetric):
  @property
  def name(self):
    return "accuracy"

  def compute(self, y_true, y_pred):
    return accuracy_score(y_true, y_pred)

class F1Metric(BaseMetric):
  @property
  def name(self):
    return "f1"

  def compute(self, y_true, y_pred):
    return f1_score(y_true, y_pred, average='weighted')

def main():


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
      {"name": "dataset_test", "type": "string"}
    ],
  }

  # Create experiment
  exp = Experiments(avro_schema=avro_schema, cross_validation=True, n_splits=5)
  exp.add_model(SKLearnModel(LogisticRegression()))
  exp.add_metric(AccuracyMetric())

  # Create the datasets
  train_1 = Dataset("train1", X_train=np.random.rand(100, 5), y_train=np.random.randint(0, 2, 100))
  test_2 = Dataset("test2", X_test=np.random.rand(75, 5), y_test=np.random.randint(0, 2, 75))

  exp.add_train(train_1)
  exp.add_test(test_2)

  # Run experiments and save results to Avro
  exp.run("examples/experiment_results.avro")

  # To view the experiment results, run this code in your terminal: "python pipeflow/read_avro"

if __name__ == "__main__":
  main()
