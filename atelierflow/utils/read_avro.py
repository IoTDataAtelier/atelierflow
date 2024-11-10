import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

def printRecord(element):
    for key, value in element.items():
        print(f"{key}: {value}")
    print("-" * 40)


class CustomPipeline:
    def __init__(self, file_path) -> None:
        self.file_path = file_path

    def run(self):
        options = PipelineOptions()
        with beam.Pipeline(options=options) as p:
            _ = (
                p
                | "Read" >> beam.io.ReadFromAvro(self.file_path)
                | "Print Records" >> beam.Map(printRecord)
            )

