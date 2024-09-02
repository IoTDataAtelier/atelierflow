from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from atelierflow import BaseModel, Experiments, BaseMetric, Dataset


class SKLearnModel(BaseModel):
  def __init__(self, model):
    # Initialize the SKLearnModel with a specific scikit-learn model.
    # The 'model' parameter is an instance of a scikit-learn model (e.g., DecisionTreeClassifier).
    self.model = model

  def fit(self, X, y):
    # This method fits the scikit-learn model to the training data.
    # 'X' is the feature matrix, and 'y' is the target vector.
    self.model.fit(X, y)

  def predict(self, X):
    # This method predicts the labels for the given feature matrix 'X' using the fitted model.
    # It returns the predicted labels.
    return self.model.predict(X)
  
  def get_parameters_description(self):
    # This function returns a dictionary containing additional model parameters that the user wants to include in the Avro schema.
    # These parameters are added to the experiment results and stored in the Avro file.
    # For example, 'learning_rate' and 'epoch' are included as key-value pairs in the returned dictionary.
    return {
      "learning_rate": "1e-10", 
      "epoch": "100"
    }


class AccuracyMetric(BaseMetric):
  @property
  def name(self):
    # This property returns the name of the metric, which is "accuracy".
    # It is used to identify the metric within the pipeline.
    return "accuracy"

  def compute(self, y_true, y_pred):
    # This method computes the accuracy score between the true labels 'y_true'
    # and the predicted labels 'y_pred'.
    # It returns the accuracy score as a float value.
    return accuracy_score(y_true, y_pred)


class F1Metric(BaseMetric):
  @property
  def name(self):
    # This property returns the name of the metric, which is "f1".
    # It is used to identify the metric within the pipeline.
    return "f1"

  def compute(self, y_true, y_pred):
    # This method computes the F1 score between the true labels 'y_true'
    # and the predicted labels 'y_pred'.
    # It returns the F1 score as a float value.
    return f1_score(y_true, y_pred, average="weighted")


def main():
  # Load the MNIST dataset
  mnist = fetch_openml("mnist_784")
  X = mnist.data
  y = mnist.target

  # Split the data into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
      {"name": "dataset_train", "type": {"type": "array", "items": "string"}},
      {"name": "dataset_test", "type": "string"},
      {"name": "learning_rate", "type": "string"},
      {"name": "epoch", "type": "string"},
    ],
  }

  # Create experiments
  exp = Experiments(avro_schema=avro_schema)

  # Add model to the experiment
  exp.add_model(SKLearnModel(DecisionTreeClassifier()))

  # Add metrics to the experiment
  exp.add_metric(AccuracyMetric())
  exp.add_metric(F1Metric())

  # Create the datasets
  train_set = Dataset("mnist_train", X_train=X_train, y_train=y_train)
  test_set = Dataset("mnist_test", X_test=X_test, y_test=y_test)

  # Add datasets to the experiment
  exp.add_train(train_set)
  exp.add_test(test_set)

  # Run experiments and save results to Avro
  exp.run("examples/experiment_results.avro")

if __name__ == "__main__":
  main()
