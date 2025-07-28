from abc import ABC, abstractmethod

class Model(ABC):
  """
  Defines the contract for any machine learning model.

  This class is a pure "component," decoupled from the pipeline logic.
  Its sole responsibility is to encapsulate an ML algorithm.
  """


  @abstractmethod
  def fit(self, X, y=None, **kwargs):
    """
    Trains the model with the provided data.
    """
    raise NotImplementedError("Subclasses must implement this method.")

  @abstractmethod
  def predict(self, X, **kwargs):
    """
    Performs predictions on new data after the model has been trained.
    """
    raise NotImplementedError("Subclasses must implement this method.")