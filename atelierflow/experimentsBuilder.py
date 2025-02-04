from fastavro import parse_schema
from atelierflow.experiments import Experiments

class ExperimentBuilder:
    def __init__(self, name):
        self.experiments = Experiments(name)
        self.model_configs = [] 
        self.metric_configs = []  

    def add_model(self, model_class, model_fit_config, **kwargs):
        self.model_configs.append({"model_class": model_class, "model_kwargs": kwargs})
        return self

    def add_metric(self, metric_class, **kwargs):
        self.metric_configs.append({"metric_class": metric_class, "metric_kwargs": kwargs})
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
        for config in self.model_configs:
            self.experiments.add_model(config["model_class"])

        # Adiciona métricas ao experimento
        for config in self.metric_configs:
            self.experiments.add_metric(config["metric_class"])

        return self.experiments, self.model_configs, self.metric_configs