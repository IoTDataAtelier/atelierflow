import os
import copy
import numpy as np
from mtsa.utils import files_train_test_split
from atelierflow.steps.step import Step
from fastavro import writer
from sklearn.model_selection import KFold
from sklearn import metrics

class LoadDataStep(Step):
    def process(self, element):
        """
        Carrega os dados de treinamento e teste do MIMII para o OSVM e
        adiciona os parâmetros de foco (kernel e nu) no dicionário de entrada.
        """
        machine_type = element['machine_type']
        train_ids = element['train_ids']
        test_id = element['test_id']
        metrics = element['metrics']
        kernel_values = element['kernel_values']
        nu_values = element['nu_values']
        
        # Carregar dados de treinamento
        X_train, y_train = self.load_data(machine_type, train_ids)
        print(f"Loaded {len(X_train)} training samples for machine_type='{machine_type}' with train_ids={train_ids}")
 
        # Carregar dados de teste
        X_test, y_test = self.load_test_data(machine_type, test_id)
        print(f"Loaded {len(X_test)} test samples for machine_type='{machine_type}' with test_id='{test_id}'")
 
        yield {
            'train_dataset': {'X': X_train, 'y': y_train},
            'test_dataset': {'X': X_test, 'y': y_test},
            'proxy_model': element['models'][0],
            'metric': metrics[0],
            'machine_type': machine_type,
            'kernel_values':kernel_values,
            'nu_values': nu_values
        }
 
    def load_data(self, machine_type, train_ids):
        X, y = [], []
        for machine_id in train_ids:
            data_path = os.path.join(os.getcwd(), "..", "..", "MIMII", machine_type, machine_id)
            if not os.path.exists(data_path):
                print(f"Warning: Data path '{data_path}' does not exist.")
                continue
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
        kernel_values = element.get('kernel_values', [model_config.model_parameters.get("kernel", 'rbf')])
        nu_values = element.get('nu_values', [model_config.model_parameters.get("nu", 0.1)])
 
        trained_models = []
        for kernel in kernel_values:
            for nu in nu_values:
                config_copy = copy.deepcopy(model_config)
                # Remover chaves que não são esperadas no construtor
                config_copy.model_parameters.pop("nu", None)
                config_copy.model_parameters.pop("kernel", None)
 
                model_instance = config_copy.model_class(**config_copy.model_parameters)
                
                model_instance.final_model.set_params(nu=nu, kernel=kernel)
 
                print(f"Training OSVM model with kernel={kernel} and nu={nu}")
                train_time = model_instance.fit(X_train, y_train)
 
                trained_models.append({
                    'model': model_instance,
                    'model_name': model_instance.__class__.__name__,
                    'kernel': kernel,
                    'nu': str(nu),
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
            model_info['AUC_ROC'] = str(auc_roc)
            print(f"Evaluated OSVM model with kernel={model_info['kernel']} and nu={model_info['nu']}, AUC_ROC={auc_roc}")
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
                "kernel": model_info.get("kernel", ""),
                "nu": model_info.get("nu", ""),
                "sampling_rate": model_info.get("sampling_rate", ""),
                "train_time_sec": model_info.get("train_time_sec", ""),
                "AUC_ROC": model_info.get("AUC_ROC", "")
            }
            results.append(result_record)
 
        mode = 'wb' if not os.path.exists(self.output_path) else 'ab'
        with open(self.output_path, mode) as out_file:
            writer(out_file, self.avro_schema, results)
 
        print(f"Appended {len(results)} records to {self.output_path}")
        yield element
 
    def name(self):
        return "AppendResultsStep"
