from dataclasses import dataclass, field
from typing import Any, Dict, Callable

@dataclass
class ModelConfig:
  model_class: Callable
  model_parameters: Dict[str, Any] = field(default_factory=dict) 
  model_fit_parameters: Dict[str, Any] = field(default_factory=dict) 