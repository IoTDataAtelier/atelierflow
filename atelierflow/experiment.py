import logging
from typing import Literal, Dict, Any, Optional
from .core.step import Step
from .core.step_result import StepResult

class Experiment:
    """
    Orchestrates the execution of a machine learning pipeline.

    This class is the main entry point for running an experiment. It takes
    a series of configured Step instances and executes them in sequence.
    """
    def __init__(self, 
        name: str,
        device: str = 'cpu',
        logging_level: Literal['NOTSET', 'DEBUG', 'INFO', 'WARNING'] = 'INFO',
        tags: Optional[Dict[str, Any]] = None,
        enable_caching: bool = False
    ):
        """
        Initializes the experiment with configuration options.

        Args:
            name (str): A descriptive name for the experiment.
            device (str): The compute device to use, e.g., 'cpu', 'cuda', 'cuda:0'.
            logging_level (str): Controls log verbosity, e.g., 'INFO', 'DEBUG'.
            tags (dict, optional): A dictionary of tags for tracking. Defaults to None.
            enable_caching (bool): If True, the framework would cache step results.
        """
       
        self.name = name
        self.steps: list[Step] = []

        self.config = {
            'device': device,
            'logging_level': logging_level.upper(),
            'tags': tags or {},
            'enable_caching': enable_caching
        }

        self._setup_logging()
    
    def _setup_logging(self):
        """Sets the global logging configuration for the experiment."""
        log_level = self.config.get('logging_level')
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        logging.info(f"Logging level set to {log_level}")

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
        
        logging.info(f"--- Starting Experiment: '{self.name}' ---")
        logging.info(f"Configuration: {self.config}")
        current_result: StepResult | None = None

        for i, step in enumerate(self.steps, 1):
            step_name = step.__class__.__name__
            logging.info(f"---> Step {i}/{len(self.steps)}: Executing '{step_name}'...")
            
            try:
                current_result = step.run(
                    input_data=current_result, 
                    experiment_config=self.config
                )
                
                if not isinstance(current_result, StepResult):
                    logging.error(f"Step '{step_name}' did not return a StepResult object.")
                    raise TypeError(f"Step '{step_name}' must return a StepResult object.")

                logging.info(f"--- Step '{step_name}' complete. ---")

            except Exception as e:
                logging.error(f"--- Step '{step_name}' failed with an error: {e}", exc_info=True)
                raise 

        logging.info(f"--- Experiment '{self.name}' Finished Successfully ---")
        return current_result

