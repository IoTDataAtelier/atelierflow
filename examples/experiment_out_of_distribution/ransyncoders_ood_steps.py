import copy
from fastavro import writer
from atelierflow.steps.step import Step

class TrainModelStep(Step):
    def process(self, element):
        model_config = element['proxy_model']
        X_train = element['train_dataset']['X']
        y_train = element['train_dataset']['y']
        batch_size_values = element.get('batch_size_values', [720])
        learning_rate_values = element.get('learning_rate_values', [1e-3])
        
        epochs_fixed = 20

        trained_models = []
        for batch_size in batch_size_values:
            for lr in learning_rate_values:
                config_copy = copy.deepcopy(model_config)
                config_copy.model_parameters.pop("batch_size", None)
                config_copy.model_parameters.pop("learning_rate", None)
                config_copy.model_parameters.pop("epochs", None)

                print(f"Training RANSynCoders model with batch_size={batch_size}, learning_rate={lr}, epochs={epochs_fixed}")
                
                model_instance = config_copy.model_class(**config_copy.model_parameters)
                model_instance.fit(X_train, None, batch_size=int(batch_size), learning_rate=lr, epochs=epochs_fixed)
                trained_models.append({
                    'model': model_instance,
                    'batch_size': str(batch_size),
                    'epoch_size': str(epochs_fixed),
                    'learning_rate': str(lr),
                    'sampling_rate': str(config_copy.model_parameters.get("sampling_rate"))
                })
        element['trained_models'] = trained_models
        yield element

    def name(self):
        return "TrainModelStep"
