import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions


class Experiments:
    def __init__(self):
        self.models = []
        self.metrics = []
        self.train_datasets = []
        self.test_datasets = []
        self.avro_schema = {}
        self.steps = []

    def add_model(self, model):
        self.models.append(model)

    def add_metric(self, metric):
        self.metrics.append(metric)

    def add_train(self, train_dataset):
        self.train_datasets.append(train_dataset)

    def add_test(self, test_dataset):
        self.test_datasets.append(test_dataset)

    def add_step(self, step):
        self.steps.append(step)

    def run(self, initial_input = None):

        print("===================================")
        print("Starting the experiment pipeline...")
        print("===================================")
        
        experiments = {
            "models": self.models,
            "train_datasets": self.train_datasets,
            "test_datasets": self.test_datasets,
            "metrics": self.metrics,
        }

        if initial_input:
            experiments.update(initial_input)

        pipeline_options = PipelineOptions()
        with beam.Pipeline(options=pipeline_options) as p:

            experiments = (
                p 
                | "Create Experiments" >> beam.Create([experiments])
            )

            for step in self.steps:  
                experiments = (
                    experiments 
                    | f"Custom Step: {step.name()}" >> beam.ParDo(step)
                )

        print("=============================")
        print("Experiment pipeline finished.")
        print("=============================")


