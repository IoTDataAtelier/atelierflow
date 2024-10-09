from abc import ABC, abstractmethod

class StepInterface(ABC):
  @abstractmethod
  def process(self, element):
    pass

  @abstractmethod
  def name(self):
    pass