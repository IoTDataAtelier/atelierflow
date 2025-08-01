import sys
from pathlib import Path
import logging
from typing import Any, Dict, Optional

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

# For this example, we use the mtsa package.

from mtsa.models import IForest
from mtsa.utils import files_train_test_split

from atelierflow.experiment import Experiment
from atelierflow.core.metric import Metric
from atelierflow.core.model import Model
from atelierflow.core.step import Step
from atelierflow.core.step_result import StepResult
from atelierflow.steps.common.save_data.save_to_avro import SaveToAvroStep

# --- Components ---
class MyIForest(Model):
  def __init__(self, **kwargs):
    self.model = IForest(**kwargs)

  def fit(self, X, y):
    self.model.fit(X, y)

  def predict(self, X):
    return self.model.predict(X)

  def score_samples(self, X):
    return self.model.score_samples(X)
  
class AucRocMetric(Metric):
  def compute(self, y_true, y_pred) -> float:
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y_true, y_pred)
  
# --- Pipeline Steps ---
class LoadAndSplitDataStep(Step):
  def __init__(self, directory: str):
    self.directory = directory

  def run(self, input_data: Optional[StepResult], experiment_config: Dict[str, Any]) -> StepResult:
    logging.info(f"Loading and splitting data from: {self.directory}")
    X_train, X_test, y_train, y_test = files_train_test_split(self.directory)
    logging.debug("Data loaded and split successfully.")
    result = StepResult()
    result.add('X_train', X_train)
    result.add('X_test', X_test)
    result.add('y_train', y_train)
    result.add('y_test', y_test)
    return result
  
class TrainModelStep(Step):
  def __init__(self, model: Model):
    self.model = model

  def run(self, input_data: Optional[StepResult], experiment_config: Dict[str, Any]) -> StepResult:
    if not input_data: 
      raise ValueError("TrainModelStep requires input data.")
    X_train, y_train = input_data.get('X_train'), input_data.get('y_train')
    logging.info(f"Fitting model '{self.model.__class__.__name__}'...")
    self.model.fit(X_train, y_train)
    logging.info("Model training complete.")
    result = StepResult()
    result.add('model', self.model)
    result.add('X_test', input_data.get('X_test'))
    result.add('y_test', input_data.get('y_test'))
    return result
  
class EvaluateModelStep(Step):
  def __init__(self, metrics: list[Metric]):
    self.metrics = metrics

  def run(self, input_data: Optional[StepResult], experiment_config: Dict[str, Any]) -> StepResult:
    if not input_data: 
      raise ValueError("EvaluateModelStep requires input data.")
    model, X_test, y_test = input_data.get('model'), input_data.get('X_test'), input_data.get('y_test')
    logging.info("Making predictions on the test set...")
    y_pred = model.score_samples(X_test)
    scores = {}
    for metric in self.metrics:
      metric_name = metric.__class__.__name__
      logging.debug(f"Calculating metric: {metric_name}")
      score = metric.compute(y_test, y_pred)
      scores[metric_name] = score
    result = StepResult()
    result.add('evaluation_scores', scores)
    return result


# --- 1. Configuration ---
DATA_DIRECTORY = "/data/henrique/CBICEmbeddedAI/cooling-fans/dB0/A2/12V/config1config5"
OUTPUT_FILE = "./anomaly_detection_results.avro"

model_component = MyIForest()  
metric_component = AucRocMetric()

# --- 2. Pipeline Assembly ---
iforest_experiment = Experiment(name="Isolation Forest Anomaly Detection", logging_level="INFO")

iforest_experiment.add_step(LoadAndSplitDataStep(directory=DATA_DIRECTORY))
iforest_experiment.add_step(TrainModelStep(model=model_component))
iforest_experiment.add_step(EvaluateModelStep(metrics=[metric_component]))

iforest_experiment.add_step(
  SaveToAvroStep(
    output_path=OUTPUT_FILE,
    data_key='evaluation_scores' # This must match the key from EvaluateModelStep
  )
)
# --- 3. Execution ---
final_results = iforest_experiment.run()

print("\n--- Final Experiment Results ---")
scores = final_results.get('evaluation_scores')
print(scores)