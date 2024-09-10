class Dataset:
  def __init__(self, name, X_train=None, y_train=None, X_test=None, y_test=None):
    self.name = name
    self.X_train = X_train
    self.y_train = y_train
    self.X_test = X_test
    self.y_test = y_test

  def has_train(self):
    return self.X_train is not None and self.y_train is not None

  def has_test(self):
    return self.X_test is not None and self.y_test is not None