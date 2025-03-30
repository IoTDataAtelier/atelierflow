import os
import copy
from atelierflow.steps.step import Step
from fastavro import writer

class TrainModelStep(Step):
    def process(self, element):
        model_config = element['proxy_model']
        X_train = element['train_dataset']['X']
        y_train = element['train_dataset']['y']
        batch_size_values = element.get('batch_size_values', [model_config.model_parameters.get("batch_size", 512)])
        learning_rate_values = element.get('learning_rate_values', [model_config.model_parameters.get("learning_rate", 1e-3)])
        
        epochs_fixed = 50

        trained_models = []
        for batch_size in batch_size_values:
            for lr in learning_rate_values:
                
                config_copy = copy.deepcopy(model_config)
                config_copy.model_parameters["batch_size"] = int(batch_size)
                config_copy.model_parameters["learning_rate"] = lr
                config_copy.model_parameters["epochs"] = epochs_fixed

                print(f"Training Hitachi model with batch_size={batch_size}, learning_rate={lr}, epochs={epochs_fixed}")
                model_instance = config_copy.model_class(**config_copy.model_parameters)
                
                train_time = model_instance.fit(X_train, y_train)
                trained_models.append({
                    'model': model_instance,
                    'batch_size': str(batch_size),
                    'epoch_size': str(epochs_fixed),
                    'learning_rate': str(lr),
                    'sampling_rate': str(config_copy.model_parameters.get("sampling_rate")),
                    'train_time_sec': str(train_time)
                })
        element['trained_models'] = trained_models
        yield element

    def name(self):
        return "TrainModelStep"
    
class AppendResultsStep(Step):
    def __init__(self, output_path, avro_schema):
        self.output_path = output_path
        self.avro_schema = avro_schema

    def process(self, element):
        results = []
        for model_info in element['trained_models']:
            result_record = {
                "batch_size": model_info.get("batch_size", ""),
                "epoch_size": model_info.get("epoch_size", ""),
                "learning_rate": model_info.get("learning_rate", ""),
                "sampling_rate": model_info.get("sampling_rate", ""),
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