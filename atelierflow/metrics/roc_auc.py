from mtsa.metrics import calculate_aucroc
from atelierflow.metrics.metric import BaseMetric

class ROCAUC(BaseMetric):
  def __init__(self, name):
    self.name = name

  def compute(self, model, X, y):
    return calculate_aucroc(model, X, y)
    