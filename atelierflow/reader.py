import os
import avro.schema
import avro.datafile
import avro.io

folder_path = "examples/isolation_forest_ood_results"

# Lista todos os arquivos .avro na pasta
avro_files = [f for f in os.listdir(folder_path) if f.endswith('.avro')]

for file_name in avro_files:
    file_path = os.path.join(folder_path, file_name)
    print(f"Lendo o arquivo: {file_name}")
    with open(file_path, "rb") as f:
        reader = avro.datafile.DataFileReader(f, avro.io.DatumReader())
        for record in reader:
            print(record)
        reader.close()
