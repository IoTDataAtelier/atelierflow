import datetime
from fastavro import writer
from niexperiments.dataset.cifar import get_cifar10_kfold_splits, get_cifar10_dataset, get_cifar10_corrupted
from niexperiments import experiments_config
from niexperiments.lib.consts import CORRUPTIONS_TYPES
from niexperiments.lib.logger import print_execution
from keras.callbacks import EarlyStopping
from niexperiments.lib.functions import filter_active
from atelierflow.steps.step import Step


class LoadDataStep(Step):
    def process(self, element):
        x_train, y_train, x_test, y_test, splits = get_cifar10_kfold_splits(experiments_config.KFOLD_N_SPLITS)
        yield {
            'x_train': x_train,
            'y_train': y_train,
            'x_test': x_test,
            'y_test': y_test,
            'splits': splits,
            'train_dataset_name': "CIFAR-10", 
            'test_dataset_name': "CIFAR-10" 
        }

    def name(self):
        return "LoadDataStep"
    
class GenerateFoldsStep(Step):
    def process(self, element):
        x_train, y_train, x_test, y_test, splits = get_cifar10_kfold_splits(experiments_config.KFOLD_N_SPLITS)
        
        for fold, (train_index, val_index) in enumerate(splits):
            yield {
                'x_train': [x_train][train_index],
                'y_train': [y_train][train_index],
                'x_val': [x_train][val_index],
                'y_val': [y_train][val_index],
                'x_test': x_test,
                'y_test': y_test,
                'fold': fold,
                'splits': splits,
                'train_dataset_name': "CIFAR-10", 
                'test_dataset_name': "CIFAR-10"
            }

    def name(self):
        return "GenerateFoldsStep"


class TrainModel(Step):
    def process(self, experiment):
        cf = filter_active(experiments_config.CONFIGS)
        for index, config in enumerate(cf):
            model_config = config['model']
            data_augmentation_layers = config['data_augmentation_layers']

            for fold, (train_index, val_index) in experiment['splits']:
                fold_number = fold + 1
                model = model_config(input_shape=experiments_config.INPUT_SHAPE)
                model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
                print_execution(fold_number, config['approach_name'], model.name)

                x_train_fold, y_train_fold = experiment['x_train'][train_index], experiment['y_train'][train_index]
                x_val_fold, y_val_fold = experiment['x_train'][val_index], experiment['y_train'][val_index]

                train_ds = get_cifar10_dataset(x_train_fold, y_train_fold, data_augmentation_layers)
                val_ds = get_cifar10_dataset(x_val_fold, y_val_fold)

                _, training_time = model.fit(
                train_ds,
                val_dataset=val_ds,
                epochs=experiments_config.EPOCHS,
                callbacks=[EarlyStopping(patience=10, monitor='val_loss', restore_best_weights=True, verbose=1)]
            )

                experiment['model'] = model
                yield experiment

    def name(self):
        return "TrainModel"


class EvaluateModel(Step):
    def process(self, experiment):
        model = experiment['model']
        test_ds = get_cifar10_dataset(experiment['x_test'], experiment['y_test'])
        loss, acc = model.evaluate(test_ds)
        
        experiment['loss'] = loss
        experiment['accuracy'] = acc

        print(f"Evaluation Results - Loss: {loss}, Accuracy: {acc}")
        yield experiment

    def name(self):
        return "EvaluateModel"


class CorruptionEvaluationStep(Step):
    def process(self, experiment):
        model = experiment['model']
        for corruption in CORRUPTIONS_TYPES:
            corrupted_dataset = get_cifar10_corrupted(corruption)
            loss, acc = model.evaluate(corrupted_dataset)
            print(f"Corruption: {corruption} - Loss: {loss}, Accuracy: {acc}")

            experiment[f'corruption_{corruption}'] = {'loss': loss, 'accuracy': acc, 'dataset_name': f"CIFAR-10 Corrupted - {corruption}"}
            yield experiment

    def name(self):
        return "CorruptionEvaluationStep"


class AppendResults(Step):
    def __init__(self, output_path, avro_schema):
        self.output_path = output_path
        self.avro_schema = avro_schema

    def process(self, experiment):
        model = experiment['model']
        metric_value = experiment['accuracy']
        metric_name = 'accuracy'

        # Resultados de corrupção, incluindo o nome do dataset Corrupted
        corruption_results = []
        for corruption in CORRUPTIONS_TYPES:
            if f'corruption_{corruption}' in experiment:
                corruption_results.append({
                    "corruption_type": corruption,
                    "corruption_loss": experiment[f'corruption_{corruption}']['loss'],
                    "corruption_accuracy": experiment[f'corruption_{corruption}']['accuracy'],
                    "corrupted_dataset_name": experiment[f'corruption_{corruption}']['dataset_name']  # Nome do dataset Corrupted
                })

        # Registro final que inclui os nomes dos datasets para treino e teste
        record = {
            "model_name": type(model).__name__,
            "metric_name": metric_name,
            "metric_value": metric_value,
            "date": datetime.datetime.now().isoformat(),
            "dataset_train": experiment['train_dataset_name'],  # Nome do dataset de treino (CIFAR-10)
            "dataset_test": experiment['test_dataset_name'],    # Nome do dataset de teste (CIFAR-10)
            "corruption_results": corruption_results            # Resultados com nomes de datasets Corrupted
        }

        # Grava o registro no arquivo Avro
        with open(self.output_path, "a+b") as out:
            writer(out, self.avro_schema, [record])
        yield experiment


    def name(self):
        return "AppendResults"
