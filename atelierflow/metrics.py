from abc import ABC, abstractmethod

class BaseMetric(ABC):
  def __init__(self, name=None, compute_params=None):
    self.name = name or self.__class__.__name__
    self.compute_params = compute_params or {}

  @abstractmethod
  def compute(self, y_true=None, y_pred=None):
    pass

  def get_compute_params(self):
    return self.compute_params
  
  def run(self, X, y=None, model=None):
    pass