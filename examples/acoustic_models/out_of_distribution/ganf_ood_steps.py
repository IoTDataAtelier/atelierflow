import apache_beam as beam
from fastavro import writer
import numpy as np
from sklearn.model_selection import KFold
import os
from mtsa.utils import files_train_test_split   

class LoadDataStep(beam.DoFn):
    def process(self, element):
        """
        Carrega os dados de treinamento e teste do MIMII conforme os parâmetros fornecidos.
        """
        machine_type = element['machine_type']
        train_ids = element['train_ids']
        test_id = element['test_id']
        metrics = element['metrics']
        batch_size_values = element['batch_size_values']
        learning_rate_values = element['learning_rate_values']

        # Construir o caminho base para a máquina específica
        base_path = os.path.join(os.getcwd(), "..", "..", "MIMII", machine_type)

        # Carregar dados de treinamento
        X_train, y_train = self.load_data(base_path, train_ids)
        print(f"Loaded {len(X_train)} training samples for machine_type='{machine_type}' with train_ids={train_ids}")

        # Carregar dados de teste
        X_test, y_test = self.load_test_data(base_path, test_id)
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
            'batch_size_values': batch_size_values,
            'learning_rate_values': learning_rate_values,
            'machine_type': machine_type 
        }

    def load_data(self, base_path, train_ids):
        """
        Carrega os dados de treinamento a partir dos IDs fornecidos.

        Retorna:
            - X_train (np.ndarray): Dados de entrada para treinamento.
            - y_train (np.ndarray): Rótulos de treinamento.
        """
        X, y = [], []
        for machine_id in train_ids:
            data_path = os.path.join(base_path, machine_id)
            if not os.path.exists(data_path):
                print(f"Warning: Data path '{data_path}' does not exist.")
                continue
            X_data, _, y_data, _ = files_train_test_split(data_path)
            print(f"Loaded {len(X_data)} samples from '{data_path}'")
            X.extend(X_data)
            y.extend(y_data)
        return np.array(X), np.array(y)

    def load_test_data(self, base_path, test_id):
        """
        Carrega os dados de teste a partir do ID fornecido.

        Retorna:
            - X_test (np.ndarray): Dados de entrada para teste.
            - y_test (np.ndarray): Rótulos de teste.
        """
        data_path = os.path.join(base_path, test_id)
        if not os.path.exists(data_path):
            print(f"Warning: Test data path '{data_path}' does not exist.")
            return np.array([]), np.array([])
        _, X_test, _, y_test = files_train_test_split(data_path)
        print(f"Loaded {len(X_test)} test samples from '{data_path}'")
        return np.array(X_test), np.array(y_test)

    def name(self):
        return "LoadDataStep"

class PrepareFoldsStep(beam.DoFn):
    def process(self, element):
        train_dataset = element['train_dataset']
        X = train_dataset['X']
        Y = train_dataset['y']

        kf = KFold(n_splits=5)
        splits = list(enumerate(kf.split(X, Y)))
        element['splits'] = splits
        yield element

    def name(self):
        return "PrepareFoldsStep"

class TrainModelStep(beam.DoFn):
    def process(self, element):
        train_dataset = element['train_dataset']
        model = element['proxy_model'].model  
        batch_size_values = element.get('batch_size_values', [32])
        learning_rate_values = element.get('learning_rate_values', [1e-3])
        splits = element['splits']
        machine_type = element['machine_type']

        for batch_size in batch_size_values:
            for learning_rate in learning_rate_values:
                print(f'\nBatch Size= {batch_size}, Learning Rate= {learning_rate}\n')
                for fold, (train_idx, val_idx) in splits:
                    print(f"Fold: {fold + 1}")
                    
                    x_train_fold = element['train_dataset']['X'][train_idx]
                    y_train_fold = element['train_dataset']['y'][train_idx]

                    model.fit(x_train_fold, y_train_fold, batch_size=int(batch_size), learning_rate=learning_rate)

                    element['sampling_rate'] = model.sampling_rate
                    element['model'] = model 
                    element['batch_size'] = batch_size
                    element['learning_rate'] = learning_rate
                    yield element

    def name(self):
        return "TrainModelStep"

class EvaluateModelStep(beam.DoFn):
    def process(self, element):
        model = element['model']
        test_dataset = element['test_dataset']
        metric = element['metric']

        paths_array = np.array(test_dataset.paths, dtype='<U90')
        auc = metric.compute(model, paths_array, test_dataset.labels)
        element['AUC_ROC'] = auc
        yield element

    def name(self):
        return "EvaluateModelStep"

class AppendResultsStep(beam.DoFn):
    def __init__(self, output_dir, avro_schema):
        self.output_dir = output_dir
        self.avro_schema = avro_schema

    def process(self, element):
        batch_size = element.get('batch_size', 'auto')
        epoch_size = '20' 
        learning_rate = element.get('learning_rate', 'auto')
        sampling_rate = element.get('sampling_rate', 'auto')
        AUC_ROCs = element.get('AUC_ROC', 'auto')
        machine_type = element.get('machine_type', 'unknown')

        print(f"  -> Appending results for model {type(element['model']).__name__} to Avro file...\n")

        record = {
            "batch_size": str(batch_size),
            "epoch_size": str(epoch_size),
            "learning_rate": str(learning_rate),
            "sampling_rate": str(sampling_rate),
            "AUC_ROCs": str(AUC_ROCs)
        }

        # Definir o caminho de saída específico para a máquina
        output_path = os.path.join(self.output_dir, f"ganf_results_{machine_type}.avro")

        # Escrever o registro no arquivo Avro correspondente
        with open(output_path, "a+b") as out:
            writer(out, self.avro_schema, [record])

    def name(self):
        return "AppendResultsStep"


