import apache_beam as beam

class Step(beam.DoFn):
  def process(self, element):
    raise NotImplementedError("Subclasses must implement this method.")

  def name(self):
    raise NotImplementedError("Subclasses must implement this method.")


