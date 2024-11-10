import avro.schema
import avro.datafile
import avro.io

# Caminho para o arquivo .avro
file_path = "examples/isolation_forest_ood_results.avro"

with open(file_path, "rb") as f:
    reader = avro.datafile.DataFileReader(f, avro.io.DatumReader())
    for record in reader:
        print(record)
    reader.close()
