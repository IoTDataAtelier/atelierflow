from abc import ABC, abstractmethod

class Metric(ABC):
  """
  Defines the contract for a performance evaluation metric.

  This is a pure "component" used by evaluation steps to produce
  a numerical score.
  """
  
  @abstractmethod
  def compute(self, y_true, y_pred, **kwargs) -> float:
    """
    Computes the metric and returns a single numerical value (float).
    """
    raise NotImplementedError("Subclasses must implement the 'compute' method.")