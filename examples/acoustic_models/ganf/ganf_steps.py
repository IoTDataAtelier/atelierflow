import apache_beam as beam
from fastavro import writer
import numpy as np
from sklearn.model_selection import KFold

from pipeflow.atelierflow.datasets.acoustic_dataset import AcousticDataset

class LoadDataStep(beam.DoFn):
    def process(self, element):
        
        train_dataset = AcousticDataset(element['path_input'], include_abnormal=False, pattern=".wav")
        test_dataset = AcousticDataset(element['path_input'], include_abnormal=True, pattern=".wav")
        
        yield {
            'train_dataset': train_dataset,
            'test_dataset': test_dataset,
            'proxy_model': element['models'][0],
            'metric': element['metrics'][0],
            'learning_rate_values': element['learning_rate_values'],
            'batch_size_values': element['batch_size_values']
        }

    def name(self):
        return "LoadDataStep"

class PrepareFoldsStep(beam.DoFn):
    def process(self, element):
        train_dataset = element['train_dataset']
        X = train_dataset.paths
        Y = train_dataset.labels

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
        
        for learning_rate in element['learning_rate_values']:
            for batch_size in element['batch_size_values']:
                print('\nlr= {}, batch= {}\n'.format(learning_rate, batch_size))
                for fold, (train_index, val_index) in element['splits']:
                    print(f"Fold: {fold + 1}")


                    x_train_fold = [train_dataset.paths[i] for i in train_index]
                    y_train_fold = [train_dataset.labels[i] for i in train_index]

                    model_copy = model.clone()
                    model_copy.fit(x_train_fold, y_train_fold, batch_size=int(batch_size), learning_rate=learning_rate)

                    element['sampling_rate'] = model.sampling_rate
                    element['model'] = model_copy
                    element['batch_size'] = batch_size
                    element['learning_rate'] = learning_rate
                    yield element
                    del model_copy

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
    def __init__(self, output_path, avro_schema):
        self.output_path = output_path
        self.avro_schema = avro_schema

    def process(self, element):
        batch_size = element['batch_size']
        epoch_size = '20'
        learning_rate = element['learning_rate']
        sampling_rate = element['sampling_rate']
        AUC_ROCs = element['AUC_ROC']
        model = element['model']

        print(f"  -> Appending results for model {type(model).__name__} to Avro file...\n")

        record = {
            "batch_size": str(batch_size),
            "epoch_size": str(epoch_size),
            "learning_rate": str(learning_rate),
            "sampling_rate": str(sampling_rate),
            "AUC_ROCs": str(AUC_ROCs)
        }
        with open(self.output_path, "a+b") as out:
            writer(out, self.avro_schema, [record])

    def name(self):
        return "AppendResultsStep"


