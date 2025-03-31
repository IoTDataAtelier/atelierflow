import os
from fastavro import reader

def ler_arquivo_avro(caminho_arquivo):
    registros = []
    with open(caminho_arquivo, 'rb') as arquivo:
        avro_reader = reader(arquivo)
        for registro in avro_reader:
            registros.append(registro)
    return registros

def ler_pacote_avro(caminho_pacote):
    dados_pacote = {}
    for raiz, _, arquivos in os.walk(caminho_pacote):
        for arquivo in arquivos:
            if arquivo.endswith('.avro'):
                caminho_arquivo = os.path.join(raiz, arquivo)
                dados = ler_arquivo_avro(caminho_arquivo)
                dados_pacote[arquivo] = dados
    return dados_pacote

if __name__ == "__main__":
    
    #caminho_arquivo = "caminho/para/seu/arquivo.avro"
    #registros = ler_arquivo_avro(caminho_arquivo)
    #print(f"Registros do arquivo {caminho_arquivo}:")
    #for reg in registros:
    #    print(reg)
    
    caminho_pacote = "experiment_ood_Hitachi"
    pacote_dados = ler_pacote_avro(caminho_pacote)
    print("\nDados do pacote de arquivos Avro:")
    for arquivo, registros in pacote_dados.items():
        print(f"\nArquivo: {arquivo}")
        for reg in registros:
            print(reg)
