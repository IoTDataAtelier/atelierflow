from fastavro import writer
import numpy as np
from sklearn.model_selection import KFold
from mtsa.utils import files_train_test_split
from atelierflow.utils.modelFactory import ModelFactory
from atelierflow.steps.step import Step

class LoadDataStep(Step):
  def process(self, element):
    X_train, X_test, y_train, y_test = files_train_test_split(element['path'])
  
    yield {
      'X_train': X_train,
      'X_test': X_test,
      'y_train': y_train,
      'y_test': y_test,
      'model_configs': element['model_configs'],
      'metric_configs': element['metric_configs'],
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
      print(model_config)
      model_class = model_config.model_class
      model_parameters = model_config.model_parameters
      model_fit_parameters = model_config.model_fit_parameters

      for fold, (train_index, val_index) in element['splits']:
        print(f"Fold: {fold + 1}")

        x_train_fold, y_train_fold = X_train[train_index], y_train[train_index]

        # Cria uma nova instância do modelo
        model = ModelFactory.create_model(model_class, **model_parameters)
        model.fit(x_train_fold, y_train_fold, **model_fit_parameters)

        element['model'] = model
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

    for metric_config in metric_configs:
      metric_class = metric_config["metric_class"]
      metric_kwargs = metric_config["metric_kwargs"]

      # Cria uma instância da métrica
      metric = metric_class(**metric_kwargs)

      # Calcula a métrica
      score = metric.compute(model, X_test, y_test)
      print(f"Score for {metric_class.__name__}: {score}")

      element[f"AUC"] = score

    yield element

  def name(self):
    return "EvaluateModelStep"

class AppendResultsStep(Step):
  def __init__(self, output_path, avro_schema):
    self.output_path = output_path
    self.avro_schema = avro_schema

  def process(self, element):
    AUC_ROCs = element['AUC']
    model = element['model']

    print(f"  -> Appending results for model {type(model).__name__} to Avro file...\n")

    record = {
      "model_name": str(type(model).__name__),
      "AUC_ROCs": str(AUC_ROCs)
    }
    with open(self.output_path, "a+b") as out:
      writer(out, self.avro_schema, [record])

    yield element

  def name(self):
    return "AppendResultsStep"


