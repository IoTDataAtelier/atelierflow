from abc import ABC, abstractmethod

class BaseMetric(ABC):
  def __init__(self):
    raise NotImplementedError("Subclasses must implement this method.")

  @abstractmethod
  def compute(self):
    raise NotImplementedError("Subclasses must implement this method.")