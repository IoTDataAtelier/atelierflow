import apache_beam as beam
import datetime
from fastavro import writer

from atelierflow.steps import StepInterface

class GenerateExperimentsStep(beam.DoFn, StepInterface):
    def process(self, experiment):
        phase = 1
        models = experiment.get('models', [])
        train_datasets = experiment.get('train_datasets', [])
        test_datasets = experiment.get('test_datasets', [])
        metrics = experiment.get('metrics', [])

        for model in models:
            for train_dataset in train_datasets:
                for test_dataset in test_datasets:
                    for metric in metrics:
                        print(f"\n[Phase {phase}] Running experiment:")
                        print(f"  Model: {type(model).__name__}")
                        print(f"  Training Dataset: {train_dataset.name}")
                        print(f"  Testing Dataset: {test_dataset.name}")
                        print(f"  Metric: {metric.name}\n")
                        phase += 1
                        x = {
                            'model': model,
                            'train_dataset': train_dataset,
                            'test_dataset': test_dataset,
                            'metric': metric
                        }
                        yield x
        


    def name(self):
        return "GenerateExperimentsStep"



class TrainModel(beam.DoFn, StepInterface):
   
    def process(self, experiment):
        model = experiment['model']
        train_dataset = experiment['train_dataset']
        model_copy = model.__class__(model=model.model)

        print(f"  -> Training model {type(model).__name__} on dataset {train_dataset.name}...")
        fit_params = model.get_fit_params()
        model_copy.fit(train_dataset.X_train, train_dataset.y_train, **fit_params)
        print("  -> Training completed.")

        experiment['model'] = model_copy

        yield experiment


    def name(self):
        return "TrainModel"



class EvaluateModel(beam.DoFn, StepInterface):
    def process(self, experiment):
        model_copy = experiment['model']
        test_dataset = experiment['test_dataset']
        metric = experiment['metric']
        print(f"  -> Evaluating model {type(model_copy).__name__} using metric {metric.name} on dataset {test_dataset.name}...")
        metric_value = metric.run(test_dataset.X_test, test_dataset.y_test, model_copy.model.model)
        print(f"  -> Result: {metric.name} = {metric_value}\n")

        experiment['metric_value'] = metric_value
        yield experiment
        

    def name(self):
        return "EvaluateModel"


class AppendResults(beam.DoFn, StepInterface):
    def __init__(self, output_path, avro_schema):
        self.output_path = output_path  
        self.avro_schema = avro_schema

    def process(self, record):


        model = record['model']
        test_dataset = record['test_dataset']
        train_dataset = record['train_dataset']
        metric_value = record['metric_value']
        metric_name = record['metric'].name

        print(f"  -> Appending results for model {type(model).__name__} with metric {metric_name} to Avro file...\n")

        record = {
            "model_name": type(model).__name__,
            "metric_name": metric_name,
            "metric_value": metric_value,
            "date": datetime.datetime.now().isoformat(),
            "dataset_test": test_dataset.name,
            "dataset_train": train_dataset.name
        }

        with open(self.output_path, "a+b") as out:
            writer(out, self.avro_schema, [record])
            


    def name(self):
        return "AppendResults"
