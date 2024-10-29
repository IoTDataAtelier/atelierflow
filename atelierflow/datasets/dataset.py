class Dataset:
  def __init__(self):
    raise NotImplementedError("Subclasses must implement this method.")
  
  def __getitem__(self, index):
    raise NotImplementedError("Subclasses must implement this method.")
    
  def __len__(self):
    raise NotImplementedError("Subclasses must implement this method.")
  

"""
Tópicos:
  curto-prazo:
    Branchs;
    Subir uma versão inicial base do atelierflow;
    Criar os Readmes;
    Ajeitar a arquitetura para subir no PIP oficial;
    encontros semanais para produzir mais no pipeflow;
    DECIDIR UM BOMMMMMM NOME (URGENTE);
    ir implementando em conjunto algumas features core


  longo-prazo:
    usar o framework não so para treinar modelo;
    feature de analise comparativa de modelo;
    feature step core (appendResult, latex);
    comentar do rumo do framework;
    marcar uma "sprint" de refatoração (próximo ano)
"""

