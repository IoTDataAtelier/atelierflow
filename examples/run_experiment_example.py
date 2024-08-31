import numpy as np

from pipeflow import BaseModel, Experiments, BaseMetric, Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score

class SKLearnModel(BaseModel):
  def __init__(self, model):
    self.model = model

  def fit(self, X, y):
    self.model.fit(X, y)

  def predict(self, X):
    return self.model.predict(X)

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
      {"name": "dataset_train", "type": {"type": "array", "items": "string"}},
      {"name": "dataset_test", "type": "string"}
    ],
  }

  # Create experiment
  exp = Experiments(avro_schema=avro_schema)
  exp.add_model(SKLearnModel(LogisticRegression()))
  exp.add_model(SKLearnModel(DecisionTreeClassifier()))
  exp.add_metric(AccuracyMetric())
  exp.add_metric(F1Metric())

  train_1 = Dataset("train1", X_train=np.random.rand(100, 5), y_train=np.random.randint(0, 2, 100))
  train_2 = Dataset("train2", X_train=np.random.rand(150, 5), y_train=np.random.randint(0, 2, 150))
  train_3 = Dataset("train3", X_train=np.random.rand(200, 5), y_train=np.random.randint(0, 2, 200))
  test_2 = Dataset("test2", X_test=np.random.rand(75, 5), y_test=np.random.randint(0, 2, 75))

  # Add multiple datasets
  exp.add_train(train_1)
  exp.add_train(train_2)
  exp.add_train(train_3)
  exp.add_test(test_2)

  # Run experiments and save results to Avro
  exp.run("pipeflow/experiment_results.avro")

  # To view the experiment results, run this code in your terminal: "python pipeflow/read_avro"

if __name__ == "__main__":
  main()
