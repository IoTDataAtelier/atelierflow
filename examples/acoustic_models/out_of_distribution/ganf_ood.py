from fastavro import parse_schema
import os
import numpy as np
from atelierflow import BaseModel, ExperimentBuilder
from mtsa.models.ganf import GANF
from examples.acoustic_models.out_of_distribution.ganf_ood_steps import (
    LoadDataStep, 
    PrepareFoldsStep, 
    TrainModelStep, 
    EvaluateModelStep, 
    AppendResultsStep
)
from atelierflow.metrics.metric import BaseMetric
from mtsa.metrics import calculate_aucroc

class GANFModel(BaseModel):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y, **kwargs):
        self.model.fit(X, y, **kwargs)

    def predict(self, X):
        return self.model.predict(X)
    
class ROCAUC(BaseMetric):
    def __init__(self):
        pass

    def compute(self, model, y, x):
        return calculate_aucroc(model, x, y)

def main():
    # Definição do Esquema Avro para salvar os resultados
    avro_schema = {
        "namespace": "example.avro",
        "type": "record",
        "name": "ModelResult",
        "fields": [
            {"name": "batch_size", "type": "string"},
            {"name": "epoch_size", "type": "string"},
            {"name": "learning_rate", "type": "string"},
            {"name": "sampling_rate", "type": "string"},
            {"name": "AUC_ROCs", "type": "string"},
        ],
    }

    # Instanciar o ExperimentBuilder
    builder = ExperimentBuilder()
    builder.set_avro_schema(avro_schema)

    # Definir os parâmetros para o modelo GANF
    batch_size_values = [1024, 512, 256, 128, 64, 32]
    learning_rate_values = [1e-9]
    sampling_rate_sound = 16000

    # Definir os parâmetros iniciais incluindo machine_type e IDs
    initial_inputs = {
        "machine_type": "slider",  
        "train_ids": ["id_02", "id_04", "id_06"],
        "test_id": "id_00",
        "batch_size_values": batch_size_values,
        "learning_rate_values": learning_rate_values
    }

    # Instanciar o modelo GANF
    ganf_model = GANFModel(model=GANF(
        sampling_rate=sampling_rate_sound, 
        mono=True, 
        use_array2mfcc=True, 
        isForWaveData=True
    ))

    # Adicionar o modelo GANF ao builder
    builder.add_model(ganf_model)

    # Adicionar a métrica AUC ROC
    builder.add_metric(ROCAUC())

    # Definir o diretório de saída para os arquivos Avro
    output_dir = "/data/joao/pipeflow/examples/ganf_ood_results"
    os.makedirs(output_dir, exist_ok=True)  

    # Adicionar os passos do pipeline
    builder.add_step(LoadDataStep())
    builder.add_step(PrepareFoldsStep())
    builder.add_step(TrainModelStep())
    builder.add_step(EvaluateModelStep())
    builder.add_step(AppendResultsStep(output_dir, parse_schema(avro_schema)))  

    # Construir o objeto Experiments
    experiments = builder.build()

    # Executar os experimentos com os inputs iniciais
    experiments.run(initial_inputs)

if __name__ == "__main__":
    main()
