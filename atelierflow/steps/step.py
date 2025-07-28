import apache_beam as beam
from abc import ABC, abstractmethod

from atelierflow.steps.step_result import StepResult

class Step(ABC):
  """
  Defines the contract for any executable step in an AtelierFlow pipeline.

  Any class that inherits from Step must implement the `run` method.
  The constructor (__init__) of the subclass is where dependencies
  (such as models, metrics, or configurations) should be injected.
  """

  @abstractmethod
  def run(self, input_data: StepResult | None) -> StepResult:
    """
    Executes the main logic of the step.

    :param input_data: The result object from the previous step.
                        Will be `None` if this is the first step in the pipeline.
    :return: A new StepResult object containing the output of this step.
    """
    raise NotImplementedError("Subclasses de Step devem implementar o método 'run'.")
