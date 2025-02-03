class ModelFactory:
  @staticmethod
  def create_model(model_class, **kwargs):
    return model_class(**kwargs)