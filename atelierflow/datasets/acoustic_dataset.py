from pathlib import Path
from atelierflow.datasets.dataset import Dataset


class AcousticDataset(Dataset):
  def __init__(self, root_dir, pattern = ".wav", include_abnormal = True, include_normal = True):
    """
    Args:
        root_dir (str): Caminho para o diretório raiz que contém os arquivos de áudio.
        include_abnormal (bool): Indica se os dados coletados devem incluir dados "anormais".
        include_normal (bool): Indica se os dados coletados devem includir dados "normais".
        pattern (str): Indica qual arquivo procurar.
        Estrutura esperada:
          root_dir/
              ├── machine_type_1/
              │   ├── id_00/
              │   │   ├── abnormal/
              │   │   └── normal/
              │   └── id_01/
              └── machine_type_2/
                  ├── id_00/
                  └── id_01/
    """
    self.paths = []  
    self.labels = [] 
    self.include_abnormal = include_abnormal
    self.include_normal = include_normal
    self.pattern = pattern  
    self._collect_files(root_dir)

  def _collect_files(self, root_dir):
    """
    Percorre o diretório root_dir para coletar paths dos arquivos de áudio
    e gerar rótulos com base nas pastas 'normal' e 'abnormal'.
    """
    for machine_dir in Path(root_dir).iterdir():
      if machine_dir.is_dir():
        for id_dir in machine_dir.iterdir():
          if id_dir.is_dir():
            
            if self.include_normal:
              normal_dir = id_dir / 'normal'
              if normal_dir.exists():
                for audio_file in normal_dir.glob(f"*{self.pattern}"):  
                  self.paths.append(str(audio_file))
                  self.labels.append(0)  

         
            if self.include_abnormal:
              abnormal_dir = id_dir / 'abnormal'
              if abnormal_dir.exists():
                for audio_file in abnormal_dir.glob(f"*{self.pattern}"):  
                  self.paths.append(str(audio_file))
                  self.labels.append(1)  

    if not self.paths:
      print(f"[Error] No matching files were found in '{root_dir}' with pattern '{self.pattern}'")

  def __getitem__(self, index):
    audio_path = self.paths[index]
    label = self.labels[index]
    return audio_path, label

  def __len__(self):
    return len(self.paths)





