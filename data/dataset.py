import numpy as np


class Dataset:
    def __init__(self, data, labels,transform=None):
        self.data_raw = data
        self._transform = transform
        self.data = self.apply_transform(self.data_raw)
        self.labels = labels
    
    def apply_transform(self, data):
        if not isinstance(self._transform, list):
            return self._transform(data) if self._transform else data
        else:
            for transform in self._transform:
                data = transform(data)
            return data

    def _update_data(self):
        '''update data when transform is changed'''
        self.data = self.apply_transform(self.data_raw)

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, new_transform):
        """ when updating transform, update data as well"""
        self._transform = new_transform
        self._update_data()

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    


# ---test---
# from transform.normalization import MeanNorm
# norm = MeanNorm(1,1)
# dataset = Dataset(np.random.rand(10,3), np.random.rand(10))
# print(len(dataset))
# print(dataset[0])
# print(dataset.transform)
# dataset.transform = norm
# print(dataset.transform)
# print(dataset[0])
# print(dataset.data_raw[0])