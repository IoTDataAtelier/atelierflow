import apache_beam as beam
from apache_beam import pvalue
from apache_beam.options.pipeline_options import PipelineOptions
from fastavro import writer, parse_schema
import datetime
from sklearn.model_selection import KFold

from atelierflow.datasets import Dataset

class Experiments:
    def __init__(self, avro_schema, cross_validation, n_splits):
        self.models = []
        self.metrics = []
        self.train_datasets = []
        self.test_datasets = []
        self.cross_validation = cross_validation
        self.n_splits = n_splits
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

        if self.cross_validation:
            experiments = self._generate_cross_validation_experiment()
        else:
            experiments = self._generate_experiments()

        print("=============================")
        print("Starting the experiment pipeline...")
        print("=============================")

        pipeline_options = PipelineOptions()
        with beam.Pipeline(options=pipeline_options) as p:
            experiments = (
                p | "Create Experiments" >> beam.Create(experiments)
            )

            experiments_results = (
                experiments | "Run Experiments" >> beam.ParDo(RunExperiment()).with_outputs(
                    "results", "errors"
                )
            )

            experiments_results.results | "Append Results to Avro" >> beam.ParDo(AppendResults(output_path, self.avro_schema))
            experiments_results.errors | "Logging Errors" >> beam.ParDo(LogErrors())

        print("=============================")
        print("Experiment pipeline finished.")
        print("=============================")

    def _generate_cross_validation_experiment(self):
        train_dataset = self.train_datasets[0]
        kf = KFold(n_splits=self.n_splits)

        for model in self.models:
            for metric in self.metrics:
                for fold_index, (train_idx, val_idx) in enumerate(kf.split(train_dataset.X_train, train_dataset.y_train)):
                    print(f"Processing fold {fold_index + 1}/{self.n_splits} with model {type(model).__name__} and metric {metric.name}.")
                    X_train_fold, X_val_fold = train_dataset.X_train[train_idx], train_dataset.X_train[val_idx]
                    y_train_fold, y_val_fold = train_dataset.y_train[train_idx], train_dataset.y_train[val_idx]

                    fold_train_dataset = Dataset(f"train_fold_{fold_index + 1}", X_train=X_train_fold, y_train=y_train_fold)
                    fold_val_dataset = Dataset(f"val_fold_{fold_index + 1}", X_test=X_val_fold, y_test=y_val_fold)

                    yield (model, fold_train_dataset, fold_val_dataset, metric)

    def _generate_experiments(self):
        phase = 1
        for model in self.models:
            for train_dataset in self.train_datasets:
                for test_dataset in self.test_datasets:
                    for metric in self.metrics:
                        print(f"\n[Phase {phase}] Running experiment:")
                        print(f"  Model: {type(model).__name__}")
                        print(f"  Training Dataset: {train_dataset.name}")
                        print(f"  Testing Dataset: {test_dataset.name}")
                        print(f"  Metric: {metric.name}\n")
                        phase += 1
                        yield (model, train_dataset, test_dataset, metric)

class LogErrors(beam.DoFn):
    def process(self, error_info):
        model_name, train_dataset, test_dataset, metric_name, e = error_info
        print("\n=============================")
        print("Error Summary")
        print(f"Model: {model_name}")
        print(f"Train Dataset: {train_dataset}")
        print(f"Test Dataset: {test_dataset}")
        print(f"Metric: {metric_name}")
        print(f"Error: {e}")
        print("=============================\n")
    

class RunExperiment(beam.DoFn):
    def process(self, experiment):
        model, train_dataset, test_dataset, metric = experiment
        model_copy = model.__class__(model=model.model)

        try:
            print(f"  -> Training model {type(model).__name__} on dataset {train_dataset.name}...")
            model_copy.fit(train_dataset.X_train, train_dataset.y_train)
            print("  -> Training completed.")

            y_pred = model_copy.predict(test_dataset.X_test)

            print(f"  -> Evaluating model {type(model).__name__} using metric {metric.name} on dataset {test_dataset.name}...")
            metric_value = metric.compute(test_dataset.y_test, y_pred)
            print(f"  -> Result: {metric.name} = {metric_value}\n")

            result_tuple = (
                type(model).__name__,
                train_dataset.name,
                test_dataset.name,
                metric.name,
                metric_value,
                model_copy
            )

            yield pvalue.TaggedOutput("results", result_tuple)
        except Exception as e:
            error_tuple = (
                type(model).__name__,
                train_dataset.name,
                test_dataset.name,
                metric.name,
                str(e) 
            )
            yield pvalue.TaggedOutput("errors", error_tuple)

class AppendResults(beam.DoFn):
    def __init__(self, output_path, avro_schema):
        self.output_path = output_path
        self.avro_schema = avro_schema

    def process(self, result):
        model_name, dataset_train, dataset_test, metric_name, metric_value, model = result
        parameters_description = model.get_parameters_description()

        print(f"  -> Appending results for model {model_name} with metric {metric_name} to Avro file...\n")

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
