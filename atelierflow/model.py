class BaseModel:
  def fit(self, X, y=None, **kwargs):
    raise NotImplementedError("Subclasses must implement this method.")

  def predict(self, X):
    raise NotImplementedError("Subclasses must implement this method.")