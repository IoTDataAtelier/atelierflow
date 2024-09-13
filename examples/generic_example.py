import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from atelierflow import BaseModel, Experiments, BaseMetric, Dataset
from sklearn.cluster import KMeans

from sklearn import metrics


# Model class for scikit-learn models
class DecisionTree(BaseModel):
    def __init__(self, model, fit_params=None, predict_params=None):
        self.model = model
        self.fit_params = fit_params or {}
        self.predict_params = predict_params or {}

    def fit(self, X, y, **kwargs):
        self.model.fit(X, y, **kwargs)

    def predict(self, X, **kwargs):
        return self.model.predict(X, **kwargs)

    def get_parameters_description(self):
        return {}

    def get_fit_params(self):
        return self.fit_params
    
    def get_predict_params(self):
        return self.predict_params
    
    def requires_supervised_data(self):
        return True


class LogisticReg(BaseModel):
    def __init__(self, model, fit_params=None, predict_params=None):
        self.model = model
        self.fit_params = fit_params or {}
        self.predict_params = predict_params or {}

    def fit(self, X, y, **kwargs):
        self.model.fit(X, y, **kwargs)

    def predict(self, X, **kwargs):
        return self.model.predict(X, **kwargs)

    def get_parameters_description(self):
        return {}

    def get_fit_params(self):
        return self.fit_params
    
    def get_predict_params(self):
        return self.predict_params
    
    def requires_supervised_data(self):
        return True
    


# F1 score metric
class F1Metric(BaseMetric):
    def __init__(self, name=None, compute_params=None):
        super().__init__(name, compute_params)

    def compute(self, y_true, y_pred):
        return f1_score(y_true, y_pred, average="weighted")
    
    def get_compute_params(self):
        return super().get_compute_params()



def main():
    # Generate synthetic data using NumPy
    X, y = np.random.rand(1000, 20), np.random.randint(0, 10, 1000)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

    # Create experiments
    exp = Experiments(avro_schema=avro_schema, cross_validation=False, n_splits=0)

    # Add models to the experiment with fit_params and predict_params
    exp.add_model(DecisionTree(DecisionTreeClassifier(), predict_params={}))
    exp.add_model(LogisticReg(LogisticRegression(), predict_params={}))

    # Add metrics to the experiment
    exp.add_metric(F1Metric(name='f1'))

    # Create the datasets
    train_set1 = Dataset("dataset_train_1", X_train=X_train, y_train=y_train)
    test_set1 = Dataset("dataset_test_1", X_test=X_test, y_test=y_test)

    # Add datasets to the experiment
    exp.add_train(train_set1)
    exp.add_test(test_set1)

    # Run experiments and save results to Avro
    exp.run("examples/experiment_results.avro")

if __name__ == "__main__":
    main()