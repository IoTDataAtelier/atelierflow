## AtelierFlow 🎨

A lightweight, flexible Python framework for creating clear, reproducible, and scalable machine learning pipelines.

AtelierFlow helps you structure your ML code into a series of modular, reusable steps. It's designed to bring clarity and standardization to your experimentation process, from data loading to model evaluation, without imposing a heavy, restrictive structure.
## ✨ Features

- Modular by Design: Build pipelines by chaining together independent, reusable `Step` components.

- Centralized Configuration: Easily manage global settings like `device` (CPU/GPU), `logging_level`, and custom `tags` for your entire experiment from one place.

- Extensible Library: Use pre-built, common steps for I/O, preprocessing, training, evaluation and others or easily create your own.

- Framework Agnostic: While providing helpers for libraries like scikit-learn, the core is lightweight and can orchestrate any Python-based ML workflow.

- Clear & Explicit: The framework prioritizes readability and explicit dependencies, making your pipelines easy to understand, share, and debug.

## 🚀 Installation

You can install the core framework via pip. Optional dependencies can be installed to add support for specific ML libraries.

```bash
# Install the core framework
pip install atelierflow

# To include support for scikit-learn based steps
pip install atelierflow[scikit-learn]

# To include support for pytorch based steps
pip install atelierflow[torch]
```

## Quick Start

Here’s how to build and run a simple pipeline in just a few lines of code. This example uses pre-built steps to generate data, train a model, evaluate it, and save the results.

```python
import logging
from sklearn.ensemble import RandomForestClassifier
from atelierflow.experiment import Experiment
from atelierflow.steps.common.save_data.save_to_avro import SaveToAvroStep

#  Steps not implemented
from atelierflow.steps.sklearn.evaluation import ClassificationEvaluationStep
from atelierflow.steps.sklearn.training import TrainModelStep

# 1. Define your components and schemas
model_component = RandomForestClassifier(n_estimators=50)
scores_schema = {'name': 'Scores', 'type': 'record', 'fields': [{'name': 'AUC', 'type': 'double'}]}

# 2. Create an Experiment and configure it
experiment = Experiment(
  name="Quick Start Classification",
  logging_level="INFO",
  tags={"project": "onboarding-example"}
)

# 3. Add steps to the pipeline
experiment.add_step(GenerateDataStep())
experiment.add_step(TrainModelStep(model=model_component))
experiment.add_step(ClassificationEvaluationStep(metrics={'AUC': roc_auc_score}))
experiment.add_step(
  SaveToAvroStep(
    output_path="./quick_start_results.avro",
    data_key='evaluation_scores',
    schema=scores_schema
  )
)

# 4. Run the experiment!
if __name__ == "__main__":
  final_results = experiment.run()
  logging.info(f"Pipeline complete. Results saved to ./quick_start_results.avro")
```


AtelierFlow is built around a few simple, powerful concepts.

- `Experiment`: The main orchestrator. You create an Experiment instance, give it a name, and provide global configurations. It is responsible for running the steps in the correct order.

- `Step`: A single, executable stage in your pipeline. A step can do anything: load data, train a model, or save a file. Every step receives the output of the previous step and the global experiment configuration.

- `StepResult`: A simple key-value store that acts as the data carrier between steps. A step adds its outputs (e.g., `result.add('trained_model', model)`) and the next step retrieves them (`input_data.get('trained_model')`).

## 💡 Working with Pre-built Steps

When using pre-built steps like `TrainModelStep` or `ClassificationEvaluationStep`, it's crucial that the objects you pass to them adhere to the framework's core interfaces.

- Models must implement the `Model` interface. Any object passed to `TrainModelStep` must be a class that inherits from `atelierflow.core.model.Model`. This ensures the step can reliably call methods like `.fit()` and `.predict()`.

- Metrics must implement the `Metric` interface. Similarly, any custom metric passed to an evaluation step must inherit from `atelierflow.core.metric.Metric` and implement the `.compute()` method.

This "programming to an interface" design is what gives AtelierFlow its flexibility. It allows the pre-built steps to work with any model or metric, as long as it follows the expected contract.

## 🛠️ Creating a Custom Step

Creating your own step is the primary way to extend AtelierFlow. It's as simple as inheriting from the `Step` base class and implementing the `run` method.

- Inherit from `Step`: Create a new class that inherits from `atelierflow.core.step.Step`.

- Use `__init__` for Configuration: Pass any parameters your step needs to its constructor.

- Implement `run`: This is where your logic goes. Use the `input_data` to get results from previous steps and use `experiment_config` to access global settings.

- Return a `StepResult`: Your step must return a `StepResult` object, even if it's empty, to continue the pipeline.

### Example: A Custom Hello World Step

```python
import logging
from atelierflow.core.step import Step
from atelierflow.core.step_result import StepResult
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class HelloWorldStep(Step):
  """A simple example of a custom step."""
  def __init__(self, message: str):
    # 1. Use __init__ for step-specific parameters
    self.message = message

  def run(self, input_data: Optional[StepResult], experiment_config: Dict[str, Any]) -> StepResult:
    # 2. Implement your logic in the 'run' method
    
    # You can access global config
    exp_name = experiment_config.get('name', 'Default Experiment')
    
    logger.info(f"Hello from the '{exp_name}' experiment!")
    logger.info(f"Custom message for this step: {self.message}")
    
    # 3. Return a StepResult
    return input_data # Pass through the data to the next step
```
## 🤝 Contributing

Contributions are welcome! Whether it's adding new pre-built steps, improving documentation, or reporting bugs, please feel free to open an issue or submit a pull request.