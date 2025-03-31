import os
import sys
import copy
import numpy as np
from fastavro import writer
from atelierflow.steps.step import Step

class LoadDataStep(Step):
    def process(self, element):
        machine_type = element['machine_type']
        train_ids = element['train_ids']
        test_id = element['test_id']
        metrics = element['metrics']
        batch_size_values = element['batch_size_values']
        learning_rate_values = element['learning_rate_values']

        # Carrega dados de treinamento
        X_train, y_train = self.load_data(machine_type, train_ids)
        print(f"Loaded {len(X_train)} training samples for machine_type='{machine_type}' with train_ids={train_ids}")

        # Carrega dados de teste
        X_test, y_test = self.load_test_data(machine_type, test_id)
        print(f"Loaded {len(X_test)} test samples for machine_type='{machine_type}' with test_id='{test_id}'")

        yield {
            'train_dataset': {'X': X_train, 'y': y_train},
            'test_dataset': {'X': X_test, 'y': y_test},
            'proxy_model': element['models'][0],
            'metric': metrics[0],
            'batch_size_values': batch_size_values,
            'learning_rate_values': learning_rate_values,
            'machine_type': machine_type 
        }

    def load_data(self, machine_type, train_ids):
        X, y = [], []
        for machine_id in train_ids:
            data_path = os.path.join(os.getcwd(), "..", "..", "MIMII", machine_type, machine_id)
            if not os.path.exists(data_path):
                print(f"Warning: Data path '{data_path}' does not exist.")
                continue
            from mtsa.utils import files_train_test_split
            X_data, _, y_data, _ = files_train_test_split(data_path)
            print(f"Loaded {len(X_data)} samples from '{data_path}'")
            X.extend(X_data)
            y.extend(y_data)
        return np.array(X), np.array(y)

    def load_test_data(self, machine_type, test_id):
        data_path = os.path.join(os.getcwd(), "..", "..", "MIMII", machine_type, test_id)
        if not os.path.exists(data_path):
            print(f"Warning: Test data path '{data_path}' does not exist.")
            return np.array([]), np.array([])
        from mtsa.utils import files_train_test_split
        _, X_test, _, y_test = files_train_test_split(data_path)
        print(f"Loaded {len(X_test)} test samples from '{data_path}'")
        return np.array(X_test), np.array(y_test)

    def name(self):
        return "LoadDataStep"
    
class TrainModelStep(Step):
    def process(self, element):
        model_config = element['proxy_model']
        X_train = element['train_dataset']['X']
        y_train = element['train_dataset']['y']
        batch_size_values = element.get('batch_size_values', [model_config.model_fit_parameters.get("batch_size", 32)])
        learning_rate_values = element.get('learning_rate_values', [model_config.model_fit_parameters.get("learning_rate", 1e-4)])
        
        epoch_size = 20
        
        trained_models = []
        for batch_size in batch_size_values:
            for lr in learning_rate_values:
                config_copy = copy.deepcopy(model_config)
                config_copy.model_fit_parameters["batch_size"] = batch_size
                config_copy.model_fit_parameters["learning_rate"] = lr
                config_copy.model_fit_parameters["epochs"] = epoch_size
                
                config_copy.model_parameters["sampling_rate"] = model_config.model_parameters.get("sampling_rate", 16000)
                print(f"Training GANF model with batch_size={batch_size}, learning_rate={lr}")
                
                model_instance = config_copy.model_class(**config_copy.model_parameters)
                train_time = model_instance.fit(X_train, y_train, **config_copy.model_fit_parameters)
                trained_models.append({
                    'model': model_instance,
                    'model_name': model_instance.__class__.__name__,
                    'batch_size': str(batch_size),
                    'epoch_size': str(epoch_size),
                    'learning_rate': str(lr),
                    'sampling_rate': str(config_copy.model_parameters.get("sampling_rate")),
                    'train_time_sec': str(train_time)
                })
        element['trained_models'] = trained_models
        yield element

    def name(self):
        return "TrainModelStep"

class EvaluateModelStep(Step):
    def process(self, element):
        metric = element['metric']
        X_test = element['test_dataset']['X']
        y_test = element['test_dataset']['y']
        for model_info in element['trained_models']:
            model_instance = model_info['model']
            auc_roc = metric.compute(model_instance, X_test, y_test)
            model_info['AUC_ROCs'] = str(auc_roc)
            print(f"Evaluated model: batch_size={model_info['batch_size']}, learning_rate={model_info['learning_rate']}, AUC_ROCs={auc_roc}")
        yield element

    def name(self):
        return "EvaluateModelStep"

class AppendResultsStep(Step):
    def __init__(self, output_path, avro_schema):
        self.output_path = output_path
        self.avro_schema = avro_schema

    def process(self, element):
        results = []
        for model_info in element['trained_models']:
            result_record = {
                "model_name": model_info.get("model_name", ""),
                "batch_size": model_info.get("batch_size", ""),
                "epoch_size": model_info.get("epoch_size", ""),
                "learning_rate": model_info.get("learning_rate", ""),
                "sampling_rate": model_info.get("sampling_rate", ""),
                "train_time_sec": model_info.get("train_time_sec", ""),
                "AUC_ROCs": model_info.get("AUC_ROCs", "")
            }
            results.append(result_record)
        mode = 'wb' if not os.path.exists(self.output_path) else 'ab'
        with open(self.output_path, mode) as out_file:
            writer(out_file, self.avro_schema, results)
        print(f"Appended {len(results)} records to {self.output_path}")
        yield element

    def name(self):
        return "AppendResultsStep"