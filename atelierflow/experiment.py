from atelierflow.steps.step import Step
from atelierflow.steps.step_result import StepResult

class Experiment:
    """
    Orchestrates the execution of a machine learning pipeline.

    This class is the main entry point for running an experiment. It takes
    a series of configured Step instances and executes them in sequence.
    """
    def __init__(self, name: str):
        """
        Initializes the experiment.

        :param name: A descriptive name for the experiment, useful for logging and tracking.
        """
        self.name = name
        self.steps: list[Step] = []

    def add_step(self, step: Step):
        """
        Adds a configured step to the experiment's pipeline.

        :param step: An instance of a class that inherits from Step.
        """
        self.steps.append(step)

    def run(self):
        """
        Executes all steps in the pipeline in the order they were added.

        The result of each step is passed as input to the next.

        :return: The final StepResult, containing the output of the last step.
        """
        if not self.steps:
            raise ValueError("Cannot run experiment: no steps have been added.")
        
        print(f"\n🚀 --- Starting Experiment: {self.name} ---")
        current_result: StepResult | None = None

        for i, step in enumerate(self.steps, 1):
            step_name = step.__class__.__name__
            print(f"\n---> ⚙️ Step {i}/{len(self.steps)}: Executing '{step_name}'...")
            
            current_result = step.run(input_data=current_result)
            
            if not isinstance(current_result, StepResult):
                raise TypeError(f"Step '{step_name}' failed to return a StepResult object.")

            print(f"---✔️ Step '{step_name}' complete.")

        print(f"\n🏁 --- Experiment '{self.name}' Finished ---")
        return current_result

