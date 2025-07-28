class StepResult:
  def __init__(self):
    self._data: dict = {}

  def add(self, key: str, value: any):
    self._data[key] = value

  def get(self, key: str) -> any:
    return self._data.get(key)