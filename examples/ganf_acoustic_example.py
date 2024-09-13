from mtsa.metrics import calculate_aucroc
from atelierflow import BaseMetric
from atelierflow import Experiments, BaseMetric, BaseModel, Dataset
from atelierflow import BaseModel
from mtsa.models.ganf import GANF
import numpy as np
import tensorflow as tf
from mtsa import files_train_test_split
path_input_1 = "sample_data/machine_type_1/id_00"
path_input_2 = "sample_data/machine_type_1/id_00"


class GANFModel(BaseModel):
    def __init__(self, model, fit_params=None, predict_params=None):
        self.model = model
        self.fit_params = fit_params or {}
        self.predict_params = predict_params or {}

    def fit(self, X, y, **kwargs):
        self.model.fit(X, y, **kwargs)

    def predict(self, X):
        return self.model.predict(X)

    def get_parameters_description(self):
        return {
            "model_version": "1.0"
        }
    
    def get_fit_params(self):
        return self.fit_params
    
    def get_predict_params(self):
        return self.predict_params
    
    def requires_supervised_data(self):
        return True


class ROCAUC(BaseMetric):
    def __init__(self, name=None, compute_params=None):
        super().__init__(name, compute_params)

    def compute(self, model, y_pred, y_true):
        return calculate_aucroc(model, y_pred, y_true)
    
    def get_compute_params(self):
        return super().get_compute_params()



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

    gpus = tf.config.experimental.list_physical_devices('GPU')
    assert len(gpus) > 0, "Not enough GPU hardware devices available"
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    X_train, X_test, y_train, y_test = files_train_test_split(path_input_1)
    if(len(y_train) == 0): 
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

    # Instantiate the Experiments
    experiments = Experiments(avro_schema=avro_schema)

    # Instantiate the GANF model
    ganf_model = GANFModel(model=GANF(mono= True, use_array2mfcc= True, isForWaveData= True))

    # Add the GANF model to the experiments
    experiments.add_model(ganf_model)

    # Add the AUC ROC metric
    experiments.add_metric(ROCAUC(name="roc_auc"))

    # Add datasets
    experiments.add_train(train_dataset)
    experiments.add_test(test_dataset)

    # Define the output path for the Avro file
    output_path = "examples/experiment_results.avro"

    # Run the experiments
    experiments.run(output_path)

if __name__ == "__main__":
    main()
