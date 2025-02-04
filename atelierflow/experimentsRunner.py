class ExperimentRunner:
  def __init__(self):
    self.experiments = []

  def add_experiment(self, experiments, model_configs, metric_configs):
    self.experiments.append((experiments, model_configs, metric_configs))

  def run_all(self, initial_input=None):
    for experiments, model_configs, metric_configs in self.experiments:
      print(f"Running experiment: {experiments.name}")

      if initial_input:
        initial_input.update({
          "model_configs": model_configs,
          "metric_configs": metric_configs,
        })

      # Executa o experimento
      experiments.run(initial_input)