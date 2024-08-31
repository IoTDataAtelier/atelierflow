from abc import ABC, abstractmethod

class BaseMetric(ABC):
  @abstractmethod
  def compute(self, y_true, y_pred):
    pass

  @property
  @abstractmethod
  def name(self):
    pass