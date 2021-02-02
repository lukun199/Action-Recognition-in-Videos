import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import h5py
from torch.utils.data import Dataset
import numpy as np


def flip_key(key):
    # c000_sadsa_000
    return key[:-3] + '9' + key[-2:]

# for CRNN
class Dataset_CRNN(Dataset):
    def __init__(self, data_path = '../dataset', split = 'train'):
        self.split = split
        self.data_path = data_path + '/processed_val/_fc_val_feats.h5'
        self.dataset = h5py.File(self.data_path,'r')
        self.orig_keys = [x for x in self.dataset.keys() if '_9' not in x]
        self.sel_keys = [self.orig_keys[x] for x in range(len(self.orig_keys)) if x%4 == 0]
        self.sel_flip_keys = [flip_key(x) for x in self.sel_keys]
        self.train_keys = [x for x in self.dataset.keys() if x not in self.sel_keys and x not in self.sel_flip_keys]
        self.val_keys = self.sel_keys
        self.num_class = len(self.dataset.keys())
        print('train_keys: ', len(self.train_keys), 'val_keys: ', len(self.val_keys), 'total_keys: ', len(self.dataset.keys()))


    def __getitem__(self, index):
        if self.split == 'train':
            train_label = np.array(int(self.train_keys[index][1:4]))
            train_data = self.dataset[self.train_keys[index]].value
        else:
            train_label = np.array(int(self.val_keys[index][1:4]))
            train_data = self.dataset[self.val_keys[index]].value

        return torch.tensor(train_data).squeeze(), torch.tensor(train_label)

    def __len__(self):
        if self.split == 'train':
            return len(self.train_keys)
        else: return len(self.val_keys)

    def get_num_class(self):
        return self.num_class


class Dataset_CRNN_VAL(Dataset):
    def __init__(self, data_path = '../dataset/', split = 'train'):
        self.split = split
        self.data_path = data_path + '/processed_val/_fc_val_feats.h5'
        self.dataset = h5py.File(self.data_path,'r')
        self.keys = [x for x in self.dataset.keys() if 'smile' not in x]
        self.idx2word = {int(x[1:4]): x[5:-4] for x in self.keys}
        print('total nums: ', len(self.keys))

    def __getitem__(self, index):
        train_label = np.array(int(self.keys[index][1:4]))
        train_data = self.dataset[self.keys[index]].value

        return torch.tensor(train_data), torch.tensor(train_label)

    def __len__(self):

        return len(self.keys)


if __name__ == '__main__':
    dataset = Dataset_CRNN('../dataset/processed_train')
