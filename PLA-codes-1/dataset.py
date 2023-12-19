import torch
import pickle
import numpy as np
import random
from torch.utils.data import Dataset
import main


def normalize(data):
    min_ = torch.min(data, dim=0).values
    max_ = torch.max(data, dim=0).values
    return (data - min_) / (max_ - min_)


# for train normal MLP
class CSIDatasetV2(Dataset):
    def __init__(self, n, name):
        with open(f'./data/{n}/{name}.pkl', 'rb') as f:
            raw_data = pickle.load(f)
        data = torch.from_numpy(raw_data)
        data = data.view(data.size(0), -1)
        self.data = normalize(data).float()
        print(f'dataset shape: {self.data.shape}')
        with open(f'./data/{n}/labels.pkl', 'rb') as f:
            raw_labels = pickle.load(f)
        labels = torch.from_numpy(raw_labels)
        self.labels = labels.view(labels.size(0), -1).long()
        with open(f'./data/{n}/timestamps.pkl', 'rb') as f:
            self.timestamps = pickle.load(f)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self,idx):
        return self.data[idx], self.labels[idx].item()


# for train siamese network
class CSIDatasetV3(Dataset):
    def __init__(self, n, name):
        with open(f'./data/{n}/{name}.pkl', 'rb') as f:
            raw_data = pickle.load(f)
        data = torch.from_numpy(raw_data)
        data = data.view(data.size(0), -1)
        self.data = normalize(data).float()
        print(f'dataset shape: {self.data.shape}')
        with open(f'./data/{n}/labels.pkl', 'rb') as f:
            raw_labels = pickle.load(f)
        labels = torch.from_numpy(raw_labels)
        self.labels = labels.view(labels.size(0), -1).long()
        self.window_size = main.config['window_size']

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self,idx):
        if idx < self.window_size:
            other_idx = random.choice(list(range(0, idx))+list(range(idx+1, self.window_size)))
        else:
            other_idx = random.choice(range(idx-self.window_size, idx))
        assert idx != other_idx
        label = 1-abs(self.labels[idx]-self.labels[other_idx])
        return self.data[idx], self.data[other_idx], label


class FeatureDatasetV1(Dataset):
    def __init__(self, n, name):
        data = []
        with open(f'./data/{n}/{name}.pkl', 'rb') as f:
            while True:
                try:
                    item = pickle.load(f)
                    data.append(item)
                except EOFError:
                    break
        data = torch.from_numpy(np.array(data))
        data = data.view(data.size(0), -1)
        self.data = normalize(data).float()
        print(f'dataset shape: {self.data.shape}')
        with open(f'./data/{n}/labels.pkl', 'rb') as f:
            raw_labels = pickle.load(f)
        idx = raw_labels.shape[0]-data.shape[0]
        labels = torch.from_numpy(raw_labels)[idx:]
        self.labels = labels.view(labels.size(0), -1).long()
        print(f'label shape: {self.labels.shape}')
        with open(f'./data/{n}/timestamps.pkl', 'rb') as f:
            self.timestamps = pickle.load(f)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self,idx):
        return self.data[idx], self.labels[idx].item()



if __name__ == '__main__':
    #dataset = CSIDatasetV4(1, 'csi')
    dataset = CSIDatasetV3(1, 'csi')
    print(dataset[0][0].shape, dataset[0][1].shape)
        