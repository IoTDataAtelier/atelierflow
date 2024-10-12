from fastavro import parse_schema
from mtsa.metrics import calculate_aucroc
from atelierflow import BaseMetric, BaseModel, Dataset, ExperimentBuilder
from mtsa.models.hitachi import Hitachi
from mtsa import files_train_test_split
from examples.mtsasteps import TrainModel, EvaluateModel, AppendResults, GenerateExperimentsStep
import os

path_input_1 = os.path.join(os.getcwd(), "examples/sample_data/machine_type_1/id_00")
path_input_2 = os.path.join(os.getcwd(), "examples/sample_data/machine_type_1/id_00")

class HitachiModel(BaseModel):
    def __init__(self, model, fit_params=None, predict_params=None):
        self.model = model
        self.fit_params = fit_params or {}
        self.predict_params = predict_params or {}

    def fit(self, X, y, **kwargs):
        self.model.fit(X, y, **kwargs)

    def predict(self, X):
        return self.model.predict(X)
  
    def get_fit_params(self):
        return self.fit_params
    
    def get_predict_params(self):
        return self.predict_params
    
    def requires_supervised_data(self):
        return True


class ROCAUC(BaseMetric):
    def __init__(self, name=None, compute_params=None):
        super().__init__(name, compute_params)

    def compute(self, X, y, model):
        return calculate_aucroc(model, X, y)
    
    def get_compute_params(self):
        return super().get_compute_params()
    
    def run(self, X, y=None, model=None):
        return self.compute(X, y, model)



def main():
    X_train, X_test, y_train, y_test = files_train_test_split(path_input_1)
    if len(y_train) == 0: 
        X_train, X_test, y_train, y_test = files_train_test_split(path_input_2)

    train_dataset = Dataset(name="synthetic_train", X_train=X_train, y_train=y_train)
    test_dataset = Dataset(name="synthetic_test", X_test=X_test, y_test=y_test)

    avro_schema = {
        "namespace": "example.avro",
        "type": "record",
        "name": "ModelResult",
        "fields": [
            {"name": "model_name", "type": "string"},
            {"name": "metric_name", "type": "string"},
            {"name": "metric_value", "type": "float"},
            {"name": "date", "type": "string"},
            {"name": "dataset_train", "type": "string"},
            {"name": "dataset_test", "type": "string"},
        ],
    }

    hitachi = HitachiModel(model=Hitachi())

    builder = ExperimentBuilder()
    builder.add_model(HitachiModel(hitachi))
    builder.set_avro_schema(avro_schema)

    builder.add_metric(ROCAUC(name="roc_auc"))

    builder.add_train_dataset(train_dataset)
    builder.add_test_dataset(test_dataset)

    output_path = "examples/experiment_results.avro"
    builder.add_step(GenerateExperimentsStep())
    builder.add_step(TrainModel())
    builder.add_step(EvaluateModel())
    builder.add_step(AppendResults(output_path, parse_schema(avro_schema)))

    experiments = builder.build()

    experiments.run()

if __name__ == "__main__":
    main()
