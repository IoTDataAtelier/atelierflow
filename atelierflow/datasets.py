import numpy as np

class Dataset:
  def __init__(self, name, X_train=None, y_train=None, X_test=None, y_test=None):
    self.name = name
    self.X_train = np.array(X_train) if X_train is not None else None
    self.y_train = np.array(y_train) if y_train is not None else None
    self.X_test = np.array(X_test) if X_test is not None else None
    self.y_test = np.array(y_test) if y_test is not None else None

    if X_train is None and X_test is None:
      raise ValueError("At least one of X_train or X_test must be provided.")
    if y_train is None and y_test is None:
      raise ValueError("At least one of y_train or y_test must be provided.")

  def has_train(self):
    return self.X_train is not None and self.y_train is not None

  def has_test(self):
    return self.X_test is not None and self.y_test is not None