from mtsa.metrics import calculate_aucroc
from pipeflow.atelierflow.metrics.metric import BaseMetric

class ROCAUC(BaseMetric):
  def __init__(self, name):
    self.name = name

  def compute(self, model, X, y):
    return calculate_aucroc(model, X, y)
    