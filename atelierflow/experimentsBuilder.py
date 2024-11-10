from fastavro import parse_schema
from atelierflow.experiments import Experiments

class ExperimentBuilder:
    def __init__(self):
        self.experiments = Experiments()

    def add_model(self, model):
        self.experiments.add_model(model)
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
    
    def set_cross_validation(self, boolean):
        self.experiments.cross_validation = boolean
        return self

    def build(self):
        return self.experiments