import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import h5py
from torch.utils.data import Dataset
import numpy as np

# for CRNN
class Dataset_CRNN(Dataset):
    def __init__(self, data_path = '../dataset'):
        self.data_path = data_path + '/processed_train/small_feats.h5'
        self.bbox_path = data_path + '/processed_train/small_bboxes.h5'
        self.dataset = h5py.File(self.data_path,'r')
        self.bbox =  h5py.File(self.bbox_path,'r')

        self.keys = list(self.dataset.keys())
        self.idx2word = {int(x[1:4]): x[5:-4] for x in self.keys}
        self.num_class = len(self.keys)


    def __getitem__(self, index):
        train_label = np.array(int(self.keys[index][1:4]))
        train_data = self.dataset[self.keys[index]].value
        train_bbox = self.bbox[self.keys[index]].value
        return torch.tensor(train_data), torch.tensor(train_label),  torch.tensor(train_bbox)

    def __len__(self):
        return len(self.keys)

    def get_num_class(self):
        return self.num_class


class Dataset_CRNN_VAL(Dataset):
    def __init__(self, data_path = '../dataset/'):
        self.data_path = data_path + '/processed_val/small_feats_val.h5'
        self.bbox_path = data_path + '/processed_val/small_bboxes_val.h5'
        self.dataset = h5py.File(self.data_path,'r')
        self.bbox = h5py.File(self.bbox_path, 'r')
        self.keys = [x for x in self.dataset.keys() if 'smile' not in x and '_9' not in x]
        self.idx2word = {int(x[1:4]): x[5:-4] for x in self.keys}
        print('total nums: ', len(self.keys))

    def __getitem__(self, index):
        train_label = np.array(int(self.keys[index][1:4]))
        train_data = self.dataset[self.keys[index]].value
        train_bbox = self.bbox[self.keys[index]].value
        return torch.tensor(train_data), torch.tensor(train_label),  torch.tensor(train_bbox)

    def __len__(self):
        return len(self.keys)





if __name__ == '__main__':
    dataset = Dataset_CRNN('../dataset/processed_train')
