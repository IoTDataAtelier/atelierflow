from fastavro import writer
import numpy as np
from sklearn.model_selection import KFold
from mtsa.mtsa.utils import files_train_test_split
from atelierflow.utils.modelFactory import ModelFactory
from atelierflow.steps.step import Step
import json

class LoadDataStep(Step):
  def process(self, element):
    for path in element['path']:
      print(f"Processing data from {path}...")
      X_train, X_test, y_train, y_test = files_train_test_split(path)
      
      yield {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'model_configs': element['model_configs'],
        'metric_configs': element['metric_configs'],
        'path': path 
      }

  def name(self):
    return "LoadDataStep"

class PrepareFoldsStep(Step):
  def process(self, element):
    X_train = element['X_train']
    y_train = element['y_train']

    kf = KFold(n_splits=5)
    splits = list(enumerate(kf.split(X_train, y_train)))
    element['splits'] = splits
    yield element

  def name(self):
    return "PrepareFoldsStep"

class TrainModelStep(Step):
  def process(self, element):
    X_train = element['X_train']
    y_train = element['y_train']
    model_configs = element['model_configs']

    for model_config in model_configs:
      print(f"Training model with config: {model_config}")
      model_class = model_config.model_class
      model_parameters = model_config.model_parameters
      model_fit_parameters = model_config.model_fit_parameters

      for fold, (train_index, val_index) in element['splits']:
        print(f"Fold: {fold + 1}")

        x_train_fold, y_train_fold = X_train[train_index], y_train[train_index]

        model = ModelFactory.create_model(model_class, **model_parameters)
        model.fit(x_train_fold, y_train_fold, **model_fit_parameters)

        element['model'] = model
        element['fold'] = fold + 1
        element['model_parameters'] = model_parameters
        element['model_fit_parameters'] = model_fit_parameters

        yield element
        del model

  def name(self):
    return "TrainModelStep"

class EvaluateModelStep(Step):
  def process(self, element):
    model = element['model']
    X_test = element['X_test']
    y_test = element['y_test']
    metric_configs = element['metric_configs']

    metrics = {}

    for metric_config in metric_configs:
      metric_class = metric_config["metric_class"]
      metric_kwargs = metric_config["metric_kwargs"]

      metric = metric_class(**metric_kwargs)
      score = metric.compute(model, X_test, y_test)

      print(f"Score for {metric_class.__name__}: {score}")
      metrics[metric_class.__name__] = str(score)

    element['metrics'] = metrics
    yield element

  def name(self):
    return "EvaluateModelStep"

class AppendResultsStep(Step):
  def __init__(self, output_path, avro_schema):
    self.output_path = output_path
    self.avro_schema = avro_schema

  def process(self, element):
    metrics = element['metrics']
    model = element['model']
    fold = element['fold']
    model_parameters = element['model_parameters']
    model_fit_parameters = element['model_fit_parameters']

    additional = {
      "fold": str(fold),
      "model_parameters": json.dumps(model_parameters),
      "model_fit_parameters": json.dumps(model_fit_parameters)
    }

    print(f"  -> Appending results for model {type(model).__name__} to Avro file...\n")

    record = {
      "model_name": str(type(model).__name__),
      "metrics": metrics,
      "additional": additional 
    }

    # Salvar no arquivo Avro
    with open(self.output_path, "a+b") as out:
      writer(out, self.avro_schema, [record])

    yield element

  def name(self):
    return "AppendResultsStep"