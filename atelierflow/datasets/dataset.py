class Dataset:
  def __init__(self):
    raise NotImplementedError("Subclasses must implement this method.")
  
  def __getitem__(self, index):
    raise NotImplementedError("Subclasses must implement this method.")
    
  def __len__(self):
    raise NotImplementedError("Subclasses must implement this method.")
