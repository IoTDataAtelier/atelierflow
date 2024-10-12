from fastavro import parse_schema
from mtsa.metrics import calculate_aucroc
from atelierflow import BaseMetric, BaseModel, Dataset, ExperimentBuilder
from mtsa.models.ransyncorders import RANSynCoders
import numpy as np
import tensorflow as tf
from mtsa import files_train_test_split
from examples.mtsasteps import TrainModel, EvaluateModel, AppendResults, GenerateExperimentsStep
import os

path_input_1 = os.path.join(os.getcwd(), "examples/sample_data/machine_type_1/id_00")
path_input_2 = os.path.join(os.getcwd(), "examples/sample_data/machine_type_1/id_00")

class RANSModel(BaseModel):
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
        # some steps
        return self.compute(X, y, model)


def generate_synthetic_dataset():
    # Parameters for synthetic data
    num_train_samples = 1000
    num_test_samples = 200
    num_features = 20
    
    # Generate random data
    X_train = np.random.rand(num_train_samples, num_features)
    y_train = np.random.randint(0, 2, num_train_samples)  # Binary labels
    X_test = np.random.rand(num_test_samples, num_features)
    y_test = np.random.randint(0, 2, num_test_samples)    # Binary labels

    # Create Dataset instances
    train_dataset = Dataset(name="synthetic_train", X_train=X_train, y_train=y_train)
    test_dataset = Dataset(name="synthetic_test", X_test=X_test, y_test=y_test)

    return train_dataset, test_dataset


def main():

    # Generate synthetic dataset or load your dataset
    # train_dataset, test_dataset = generate_synthetic_dataset()

    X_train, X_test, y_train, y_test = files_train_test_split(path_input_1)
    if len(y_train) == 0: 
        X_train, X_test, y_train, y_test = files_train_test_split(path_input_2)

    train_dataset = Dataset(name="synthetic_train", X_train=X_train, y_train=y_train)
    test_dataset = Dataset(name="synthetic_test", X_test=X_test, y_test=y_test)

    # Define the Avro schema for saving results
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

    # Instantiate the ExperimentBuilder
    builder = ExperimentBuilder()
    builder.set_avro_schema(avro_schema)

    # Instantiate the RANS model
    rans_model = RANSModel(model=RANSynCoders(is_acoustic_data=True, mono=True, normal_classifier=1, abnormal_classifier=0, synchronize=True), fit_params={"epochs": 5})

    # Add the RANS model to the builder
    builder.add_model(rans_model)

    # Add the AUC ROC metric
    builder.add_metric(ROCAUC(name="roc_auc"))

    # Add datasets
    builder.add_train_dataset(train_dataset)
    builder.add_test_dataset(test_dataset)

    # Define the output path for the Avro file
    output_path = "examples/experiment_results.avro"
    builder.add_step(GenerateExperimentsStep())
    builder.add_step(TrainModel())
    builder.add_step(EvaluateModel())
    builder.add_step(AppendResults(output_path, parse_schema(avro_schema)))

    # Build the experiments object
    experiments = builder.build()

    # Run the experiments
    experiments.run()

if __name__ == "__main__":
    main()
