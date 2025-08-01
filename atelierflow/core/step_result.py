from typing import Any

class StepResult:
  def __init__(self):
    self._data = {}

  def add(self, key: str, value: Any):
    self._data[key] = value

  def get(self, key: str, default: Any = None) -> Any:
    return self._data.get(key, default)