from dataclasses import dataclass
from typing import Any, Dict, Callable

@dataclass
class ModelConfig:
  model_class: Callable
  model_parameters: Dict[str, Any]
  model_fit_parameters: Dict[str, Any]