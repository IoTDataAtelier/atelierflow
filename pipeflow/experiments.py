import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from fastavro import writer, parse_schema
import datetime

class ExperimentResult:
  def __init__(self, model_name, train_dataset_names, test_dataset_name, metrics):
    self.model_name = model_name
    self.train_dataset_names = train_dataset_names
    self.test_dataset_name = test_dataset_name
    self.metrics = metrics

  def to_dict(self):
    return {
      "model_name": self.model_name,
      "train_dataset_names": self.train_dataset_names,
      "test_dataset_name": self.test_dataset_name,
      "metrics": self.metrics
    }

class Experiments:
  def __init__(self, avro_schema):
    self.models = []
    self.metrics = []
    self.train_datasets = []
    self.test_datasets = []
    self.avro_schema = parse_schema(avro_schema)

  def add_model(self, model):
    self.models.append(model)

  def add_metric(self, metric):
    self.metrics.append(metric)

  def add_train(self, train_dataset):
    self.train_datasets.append(train_dataset)

  def add_test(self, test_dataset):
    self.test_datasets.append(test_dataset)

  def run(self, output_path):
    if not self.models:
      raise ValueError("At least one model must be added.")
    if not self.metrics:
      raise ValueError("At least one metric must be added.")
    if not self.train_datasets:
      raise ValueError("At least one training dataset must be added.")
    if not self.test_datasets:
      raise ValueError("At least one testing dataset must be added.")

    pipeline_options = PipelineOptions()
    with beam.Pipeline(options=pipeline_options) as p:
      _ = (
        p
        | "Create Experiments" >> beam.Create(self._generate_experiments())
        | "Run Experiments" >> beam.ParDo(RunExperiment())
        | "Append Results to Avro" >> beam.ParDo(AppendResults(output_path, self.avro_schema))
      )

  def _generate_experiments(self):
    for model in self.models:
      yield (model, self.train_datasets, self.test_datasets, self.metrics)              

class RunExperiment(beam.DoFn):
  def process(self, experiment):
    model, train_datasets, test_datasets, metrics = experiment
    model_copy = model.__class__(model.model)

    train_dataset_names = [ds.name for ds in train_datasets if ds.has_train()]
    for train_dataset in train_datasets:
      if train_dataset.has_train():
        model_copy.fit(train_dataset.X_train, train_dataset.y_train)

    for test_dataset in test_datasets:
      if test_dataset.has_test():
        y_pred = model_copy.predict(test_dataset.X_test)
        results = {}
        for metric in metrics:
          results[metric.name] = metric.compute(test_dataset.y_test, y_pred)

        yield ExperimentResult(
          model_name=type(model).__name__,
          train_dataset_names=train_dataset_names,
          test_dataset_name=test_dataset.name,
          metrics=results
        )

class AppendResults(beam.DoFn):
  def __init__(self, output_path, avro_schema):
    self.output_path = output_path
    self.avro_schema = avro_schema

  def process(self, result):
    for metric_name, metric_value in result.metrics.items():
      record = {
        "model_name": result.model_name,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "model_version": "1.0",
        "date": datetime.datetime.now().isoformat(),
        "dataset_train": result.train_dataset_names,
        "dataset_test": result.test_dataset_name,
      }

      with open(self.output_path, "a+b") as out:
        writer(out, self.avro_schema, [record])

  