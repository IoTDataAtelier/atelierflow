import os
import copy
import numpy as np
from mtsa.utils import files_train_test_split
from atelierflow.steps.step import Step
from fastavro import writer
from sklearn.model_selection import KFold 

class LoadDataStep(Step):
    def process(self, element):
         """
         Carrega os dados de treinamento e teste do MIMII conforme os parâmetros fornecidos.
         """
         machine_type = element['machine_type']
         train_ids = element['train_ids']
         test_id = element['test_id']
         metrics = element['metrics']
         max_features_values = element['max_features_values']
         n_estimators_values = element['n_estimators_values']
 
         # Carregar dados de treinamento
         X_train, y_train = self.load_data(machine_type, train_ids)
         print(f"Loaded {len(X_train)} training samples for machine_type='{machine_type}' with train_ids={train_ids}")
 
         # Carregar dados de teste
         X_test, y_test = self.load_test_data(machine_type, test_id)
         print(f"Loaded {len(X_test)} test samples for machine_type='{machine_type}' with test_id='{test_id}'")
 
         yield {
             'train_dataset': {
                 'X': X_train,
                 'y': y_train
             },
             'test_dataset': {
                 'X': X_test,
                 'y': y_test
             },
             'proxy_model': element['models'][0],
             'metric': metrics[0],
             'max_features_values': max_features_values,
             'n_estimators_values': n_estimators_values,
             'machine_type': machine_type 
         }
 
    def load_data(self, machine_type, train_ids):
         """
         Carrega os dados de treinamento a partir dos IDs fornecidos.
         """
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
         """
         Carrega os dados de teste a partir do ID fornecido.
         """
         data_path = os.path.join(os.getcwd(), "..", "..", "MIMII", machine_type, test_id)
         if not os.path.exists(data_path):
             print(f"Warning: Test data path '{data_path}' does not exist.")
             return np.array([]), np.array([])
         _, X_test, _, y_test = files_train_test_split(data_path)
         print(f"Loaded {len(X_test)} test samples from '{data_path}'")
         return np.array(X_test), np.array(y_test)

    def name(self):
        return "LoadDataStep"

class PrepareFoldsStep(Step):
     def process(self, element):
         train_dataset = element['train_dataset']
         X = train_dataset['X']
         Y = train_dataset['y']
 
         kf = KFold(n_splits=5, shuffle=True, random_state=1)
         splits = list(enumerate(kf.split(X, Y)))
         element['splits'] = splits
         yield element
 
     def name(self):
         return "PrepareFoldsStep"
     
class TrainModelStep(Step):
    def process(self, element):
        model_config = element['proxy_model']
        X_train = element['train_dataset']['X']
        y_train = element['train_dataset']['y']
        max_features_values = element.get('max_features_values', [model_config.model_parameters.get("max_features", 1.0)])
        n_estimators_values = element.get('n_estimators_values', [model_config.model_parameters.get("n_estimators", 100)])
        
        trained_models = []
        for max_features in max_features_values:
            for n_estimators in n_estimators_values:
                config_copy = copy.deepcopy(model_config)
                config_copy.model_parameters["max_features"] = max_features
                config_copy.model_parameters["n_estimators"] = n_estimators
                
                model_instance = config_copy.model_class(**config_copy.model_parameters)
                print(f"Training model with n_estimators={n_estimators}, max_features={max_features}")
                train_time = model_instance.fit(X_train, y_train)
                trained_models.append({
                    'model': model_instance,
                    'n_estimators': str(n_estimators),
                    'max_features': str(max_features),
                    'max_samples': str(config_copy.model_parameters.get("max_samples")),
                    'contamination': str(config_copy.model_parameters.get("contamination")),
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
            print(f"Evaluated model: n_estimators={model_info['n_estimators']}, max_features={model_info['max_features']}, AUC_ROC={auc_roc}")
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
                "n_estimators": model_info.get("n_estimators", ""),
                "max_features": model_info.get("max_features", ""),
                "contamination": model_info.get("contamination", ""),
                "max_samples": model_info.get("max_samples", ""),
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
