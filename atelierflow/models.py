from abc import ABC, abstractmethod

class BaseModel(ABC):
  @abstractmethod
  def fit(self, X=None, y=None, **kwargs):
    pass

  @abstractmethod
  def predict(self, X, **kwargs):
    pass

  def get_fit_params(self):
    return self.fit_params

  def get_predict_params(self):
    return self.predict_params