import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from fastavro import writer, parse_schema
import datetime

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

  def show_metrics(self):
    return self.metrics

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
      for train_dataset in self.train_datasets:
        for test_dataset in self.test_datasets:
          for metric in self.metrics:
            yield (model, train_dataset, test_dataset, metric)

class RunExperiment(beam.DoFn):
  def process(self, experiment):

    model, train_dataset, test_dataset, metric = experiment
    model_copy = model.__class__(model=model.model)

    model_copy.fit(train_dataset.X_train, train_dataset.y_train)

    y_pred = model_copy.predict(test_dataset.X_test)
    
    metric_value = metric.compute(test_dataset.y_test, y_pred)

    yield (
      type(model).__name__,
      train_dataset.name,
      test_dataset.name,
      metric.name,
      metric_value,
      model_copy,
    )

class AppendResults(beam.DoFn):
  def __init__(self, output_path, avro_schema):
    self.output_path = output_path
    self.avro_schema = avro_schema

  def process(self, result):
    model_name, dataset_train, dataset_test, metric_name, metric_value, model = result
    parameters_description = model.get_parameters_description()
     
    record = {
      "model_name": model_name,
      "metric_name": metric_name,
      "metric_value": metric_value,
      "date": datetime.datetime.now().isoformat(),
      "dataset_train": dataset_train,
      "dataset_test": dataset_test,
    }

    record.update(parameters_description)

    with open(self.output_path, "a+b") as out:
      writer(out, self.avro_schema, [record])

  