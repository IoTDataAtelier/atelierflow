import logging
import os
from typing import Any, Dict, Optional
import pandas as pd
from fastavro import reader, writer, parse_schema

from atelierflow.core.step import Step
from atelierflow.core.step_result import StepResult

logger = logging.getLogger(__name__)

class SaveToAvroStep(Step):
  """
  A generic step to save data into an Avro file, with support for
  overwriting or appending.
  """
  def __init__(
      self,
      output_path: str,
      data_key: str,
      schema: Optional[Dict[str, Any]] = None,
      append: bool = True
  ):
    """
    Initializes the step.

    Args:
      output_path (str): The full path where the .avro file will be saved.
      data_key (str): The key of the data to save from the previous step's result.
      schema (dict, optional): A valid Avro schema.
      append (bool): If True, appends to the file. If False, overwrites. Defaults to True.
    """
    self.output_path = output_path
    self.data_key = data_key
    self.append = append
    self.parsed_schema = parse_schema(schema) if schema else None

    if self.append and not self.parsed_schema:
      raise ValueError(
        "A 'schema' must be provided when using append mode (append=True) "
        "to ensure data consistency."
      )

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
      logger.warning(f"No records found for key '{self.data_key}'. Nothing to save.")
      return input_data

    schema_to_use = self.parsed_schema
    file_exists = os.path.exists(self.output_path)
    mode = 'a+b' if self.append and file_exists else 'wb'

    if not schema_to_use:
      logger.debug("Inferring schema for new file.")
      inferred_schema_dict = {
        'doc': 'Schema inferred by AtelierFlow', 'name': 'InferredRecord', 'type': 'record',
        'fields': [{'name': k, 'type': ['null', self._infer_avro_type(v)]} for k, v in records[0].items()],
      }
      schema_to_use = parse_schema(inferred_schema_dict)

    try:
      with open(self.output_path, mode) as out:
        writer(out, schema_to_use, records)

      if mode == 'a+b':
        logger.info(f"Successfully appended {len(records)} records to {self.output_path}")
      else:
        logger.info(f"Successfully saved {len(records)} records to new file: {self.output_path}")
    except Exception as e:
        logger.error(f"Failed to write Avro file at {self.output_path}: {e}")
        raise

    return input_data

  def _infer_avro_type(self, value: Any) -> str:
    if isinstance(value, int): return 'long'
    if isinstance(value, float): return 'double'
    if isinstance(value, bool): return 'boolean'
    return 'string'
