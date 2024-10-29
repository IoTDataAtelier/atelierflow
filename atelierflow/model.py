class BaseModel:
  def fit(self):
    raise NotImplementedError("Subclasses must implement this method.")

  def predict(self):
    raise NotImplementedError("Subclasses must implement this method.")