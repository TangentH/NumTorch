import numpy as np

class Dataloader:
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.idx = 0
        self.data = self.dataset.data
        self.labels = self.dataset.labels
        self.size = len(self.dataset)
        self.indices = np.arange(self.size)
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __iter__(self):
        '''call every time when entering for loop (epoch)'''
        # print("__iter__")
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self
    
    def __next__(self):
        # print("__next__")
        if self.idx >= self.size:
            self.idx = 0
            raise StopIteration
        else:
            start = self.idx
            end = min(self.idx + self.batch_size, self.size)
            self.idx = end
            return self.data[self.indices[start:end]], self.labels[self.indices[start:end]]
    
    def __len__(self):
        return self.size//self.batch_size + 1 if self.n % self.batch_size != 0 else self.size//self.batch_size
    

# ---test---
# from data.dataset import Dataset
# data = np.random.rand(10,3)
# labels = np.random.rand(10)
# dataset = Dataset(data,labels)
# dataloader = Dataloader(dataset, batch_size=3, shuffle=True)
# for epoch in range(2):
#     for i, (x,y) in enumerate(dataloader):
#         print(f"batch {i}:")
#         print(f"x: {x}")
#         print(f"y: {y}")
#         print()