import numpy as np

class DataLoader:
    def __init__(self, dataset, batch_size=8, shuffle=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(dataset))  
        self.current_idx = 0                   

    def __len__(self):
        return int(np.ceil(len(self.dataset) / self.batch_size))

    def __iter__(self):
        self.current_idx = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self

    def __next__(self):
        if self.current_idx >= len(self.dataset):
            raise StopIteration
        
        batch_indices = self.indices[self.current_idx:self.current_idx + self.batch_size]
        batch = [self.dataset[i] for i in batch_indices] 

        self.current_idx += self.batch_size
        
        X_batch, y_batch = zip(*batch)
        
        return np.array(X_batch), np.array(y_batch)


