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


class PrepareFoldsStep(beam.DoFn):
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

class TrainModelStep(beam.DoFn):
    def process(self, element):

        model_wrapper = element['proxy_model'] 

        max_features_values = element.get('max_features_values', [1.0])
        n_estimators_values = element.get('n_estimators_values', [100])

        for max_features in max_features_values:
            for n_estimators in n_estimators_values:
                print(f'\nmax_features= {max_features}, n_estimators= {n_estimators}\n')
                for fold, (train_index, val_index) in element['splits']:
                    print(f"Fold: {fold + 1}")

                    X_train_fold = element['train_dataset']['X'][train_index]
                    y_train_fold = element['train_dataset']['y'][train_index]

                    # Criar uma nova instância do modelo com os parâmetros atuais
                    new_model_wrapper = model_wrapper.create_new_instance(
                        n_estimators=n_estimators,
                        max_features=max_features,
                        contamination=model_wrapper.model.contamination,
                        max_samples=model_wrapper.model.max_samples,
                        sampling_rate=model_wrapper.model.sampling_rate
                    )

                    new_model_wrapper.fit(X_train_fold, y_train_fold)

                    element['sampling_rate'] = new_model_wrapper.model.sampling_rate
                    element['model'] = new_model_wrapper.model
                    element['max_features'] = max_features
                    element['n_estimators'] = n_estimators
                    element['contamination'] = new_model_wrapper.model.contamination  
                    element['max_samples'] = new_model_wrapper.model.max_samples
                    yield element

        return

    def name(self):
        return "TrainModelStep"


class EvaluateModelStep(beam.DoFn):
    def process(self, element):
        model = element['model']
        test_dataset = element['test_dataset']
        metric = element['metric']

        X_test = np.array(test_dataset['X'])
        y_test = np.array(test_dataset['y'])

        auc = metric.compute(model, X_test, y_test)
        element['AUC_ROC'] = auc
        yield element

    def name(self):
        return "EvaluateModelStep"
    
class AssignKeyStep(beam.DoFn):
    def process(self, element):
        """
        Atribui uma chave única para cada configuração de experimento baseada em um conjunto de parâmetros.
        """
        n_estimators = element.get('n_estimators', 'auto')
        max_features = element.get('max_features', 'auto')
        contamination = element.get('contamination', 'auto')
        max_samples = element.get('max_samples', 'auto')
        sampling_rate = element.get('sampling_rate', 'auto')

        key = (n_estimators, max_features, contamination, max_samples, sampling_rate)

        yield (key, element['AUC_ROC'])
    
    def name(self):
        return "AssignKeyStep"

    
class CalculateConfidenceIntervalStep(beam.DoFn):
    def __init__(self, num_bootstrap=1000, confidence_level=0.95):
        self.num_bootstrap = num_bootstrap
        self.confidence_level = confidence_level

    def process(self, element):
        """
        Recebe um par (chave, lista de AUC_ROC).
        Calcula o intervalo de confiança de 95% usando Bootstrap.
        
        Retorna um dicionário com lower_bound, mean_auc, e upper_bound.
        """
        key, auc_scores = element
        n_estimators, max_features, contamination, max_samples, sampling_rate = key

        if not auc_scores:
            print(f"Sem scores AUC_ROC disponíveis para configuração: n_estimators={n_estimators}, max_features={max_features}, contamination={contamination}, max_samples={max_samples}, sampling_rate={sampling_rate}.")
            return

        bootstrap_means = []
        for _ in range(self.num_bootstrap):
            sample = np.random.choice(auc_scores, size=len(auc_scores), replace=True)
            bootstrap_means.append(np.mean(sample))

        lower_bound = np.percentile(bootstrap_means, (1 - self.confidence_level) / 2 * 100)
        upper_bound = np.percentile(bootstrap_means, (1 + self.confidence_level) / 2 * 100)
        mean_auc = np.mean(auc_scores)

        confidence_record = {
            "n_estimators": str(n_estimators),
            "max_features": str(max_features),
            "contamination": str(contamination),
            "max_samples": str(max_samples),
            "sampling_rate": str(sampling_rate),
            "lower_bound": lower_bound,
            "mean_auc": mean_auc,
            "upper_bound": upper_bound
        }

        yield confidence_record

    def name(self):
        return "CalculateConfidenceIntervalStep"

class AppendResultsStep(beam.DoFn):
    def __init__(self, output_dir, avro_schema):
        self.output_dir = output_dir
        self.avro_schema = avro_schema

    def process(self, element):
        """
        Processa os registros de confiança e os salva em arquivos Avro separados por configuração.
        """
        n_estimators = element.get('n_estimators', 'auto')
        max_features = element.get('max_features', 'auto')
        contamination = element.get('contamination', 'auto')
        max_samples = element.get('max_samples', 'auto')
        sampling_rate = element.get('sampling_rate', 'auto')
        lower_bound = element.get('lower_bound', 'auto')
        mean_auc = element.get('mean_auc', 'auto')
        upper_bound = element.get('upper_bound', 'auto')

        print(f"  -> Salvando resultados de confiança para configuração: n_estimators={n_estimators}, max_features={max_features}, contamination={contamination}, max_samples={max_samples}, sampling_rate={sampling_rate} no arquivo Avro...\n")

        record = {
            "n_estimators": str(n_estimators),
            "max_features": str(max_features),
            "contamination": str(contamination),
            "max_samples": str(max_samples),
            "sampling_rate": str(sampling_rate),
            "lower_bound": float(lower_bound),
            "mean_auc": float(mean_auc),
            "upper_bound": float(upper_bound)
        }

        # Definir o caminho de saída específico para a configuração
        config_identifier = f"n_estimators_{n_estimators}_max_features_{max_features}_contamination_{contamination}_max_samples_{max_samples}_sampling_rate_{sampling_rate}"
        output_path = os.path.join(self.output_dir, f"isolation_forest_confidence_results_{config_identifier}.avro")

        # Escrever o registro no arquivo Avro correspondente
        with open(output_path, "a+b") as out:
            writer(out, self.avro_schema, [record])

    def name(self):
        return "AppendResultsStep"




