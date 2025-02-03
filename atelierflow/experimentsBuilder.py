from fastavro import parse_schema
from atelierflow.experiments import Experiments

class ExperimentBuilder:
    def __init__(self):
        self.experiments = Experiments()
        self.model_kwargs = {}

    def add_model(self, model_class, **kwargs):
        self.experiments.add_model(model_class)
        self.model_kwargs = kwargs
        return self

    def add_metric(self, metric):
        self.experiments.add_metric(metric)
        return self

    def add_train_dataset(self, train_dataset):
        self.experiments.add_train(train_dataset)
        return self

    def add_test_dataset(self, test_dataset):
        self.experiments.add_test(test_dataset)
        return self

    def add_step(self, step):
        self.experiments.add_step(step)
        return self

    def set_avro_schema(self, avro_schema):
        self.experiments.avro_schema = parse_schema(avro_schema)
        return self
    
    def build(self):
        return self.experiments, self.model_kwargs