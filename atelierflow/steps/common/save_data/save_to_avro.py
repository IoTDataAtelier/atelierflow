import logging
from typing import Any, Dict, Optional
import pandas as pd
from fastavro import writer, parse_schema

from atelierflow.core.step import Step
from atelierflow.core.step_result import StepResult

logger = logging.getLogger(__name__)

class SaveToAvroStep(Step):
  """
  A generic step to save data from a previous step into an Avro file,
  allowing for a user-defined or inferred schema.
  """
  def __init__(self, output_path: str, data_key: str, schema: Optional[Dict[str, Any]] = None):
    """
    Initializes the step.

    Args:
      output_path (str): The full path where the .avro file will be saved.
      data_key (str): The key corresponding to the data to be saved from the
                      previous step's StepResult.
      schema (dict, optional): A valid Avro schema dictionary. If not provided,
                                a basic schema will be inferred from the data.
                                Defaults to None.
    """
    self.output_path = output_path
    self.data_key = data_key
    self.parsed_schema = parse_schema(schema) if schema else None

  def run(self, input_data: Optional[StepResult], experiment_config: Dict[str, Any]) -> StepResult:
    if not input_data:
      raise ValueError("SaveToAvroStep requires input data from a previous step.")

    data_to_save = input_data.get(self.data_key)
    if data_to_save is None:
      raise ValueError(f"Data key '{self.data_key}' not found in the previous StepResult.")

    if isinstance(data_to_save, pd.DataFrame):
      records = data_to_save.to_dict('records')
    elif isinstance(data_to_save, list) and all(isinstance(i, dict) for i in data_to_save):
        records = data_to_save
    elif isinstance(data_to_save, dict):
      records = [data_to_save]
    else:
      raise TypeError(f"Data for key '{self.data_key}' must be a pandas DataFrame, a list of dicts, or a single dict.")

    if not records:
      logger.warning(f"No records found for key '{self.data_key}'. Avro file will be empty.")
      return input_data

    if self.parsed_schema:
      schema_to_use = self.parsed_schema
      logger.debug("Using user-provided Avro schema.")
    else:
      logger.debug("No schema provided, inferring a basic schema from the data.")
      inferred_schema_dict = {
        'doc': 'Schema inferred from data by AtelierFlow',
        'name': 'InferredRecord',
        'type': 'record',
        'fields': [{'name': key, 'type': ['null', self._infer_avro_type(value)]} for key, value in records[0].items()],
      }
      schema_to_use = parse_schema(inferred_schema_dict)

    try:
      with open(self.output_path, 'wb') as out:
        writer(out, schema_to_use, records)
      logger.info(f"Successfully saved data to Avro file: {self.output_path}")
    except Exception as e:
      logger.error(f"Failed to write Avro file at {self.output_path}: {e}")
      raise

    return input_data

  def _infer_avro_type(self, value: Any) -> str:
    """A simple helper to infer Avro type from Python type."""
    if isinstance(value, int):
      return 'long'
    if isinstance(value, float):
      return 'double'
    if isinstance(value, bool):
      return 'boolean'
    return 'string'
