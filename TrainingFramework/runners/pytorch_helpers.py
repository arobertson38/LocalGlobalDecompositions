"""
Hello!
This file contains useful code for training torch models. Code such as
custom data loaders. 

Created by: Andreas E. Robertson
Contact: arobertson38@gatech.edu
"""
import torch
import h5py

# train test split functions

def train_test_split_h5py(filename, data_key='micros', fraction=0.9):
    """ produces a training and a testing data set. 

    As a word of caution, this is specialized specifically
    for the code that I have currently written. 

    """
    data = h5py.File(filename, 'r')

    # randomly splitting the data
    from random import choice
    order = torch.randperm(len(data[data_key]))
    cutoff = int(len(data[data_key]) * fraction)
    train_indx = order[:cutoff]
    test_indx = order[cutoff:]

    test_dataset = h5pyMicroDatasetTrainTest(data, indxs=test_indx)
    train_dataset = h5pyMicroDatasetTrainTest(data, indxs=train_indx)

    return train_dataset, test_dataset


def train_test_split(data, fraction=0.8):
    from random import choice
    order = torch.randperm(len(data))
    cutoff = int(len(data) * fraction)
    train = data[order[:cutoff], ...]
    test = data[order[cutoff:], ...]
    return train, test

# custom dataset classes

class h5pyMicroDataset(torch.utils.data.Dataset):
    """
    A class for loading h5py data so that it can be indexed
    without decompressing everything. 

    This is really only valuable if the chunk size is [1, ...].
    """
    def __init__(self, data, *args, data_key='micros', dtype=torch.float):
        """
        Accepts data as both a string and a h5py file. 
        """
        if type(data) is str:
            self.data = h5py.File(data, 'r')
        elif type(data) is h5py.File:
            self.data = data
        else:
            raise NotImplementedError(f"data of type {type(data)} is not supported.")

        self.data_key = data_key
        self.dtype = dtype
        
        if len(args) > 0:
            for arg in args:
                assert len(arg) == len(self.data[self.data_key]), \
                        "All given data must have the same length."

        self.args = args

    def __len__(self):
        return len(self.data[self.data_key])

    def __getitem__(self, indx):
        if torch.is_tensor(indx):
            indx = indx.tolist()
        
        return torch.from_numpy( \
            self.data[self.data_key][indx, ...]).type(self.dtype), \
            *[arg[indx] for arg in self.args]

class h5pyMicroDatasetTrainTest(torch.utils.data.Dataset):
    """
    A class for loading h5py data so that it can be indexed
    without decompressing everything. 

    This is really only valuable if the chunk size is [1, ...].
    """
    def __init__(self, data, *args, indxs, data_key='micros', dtype=torch.float):
        """
        Accepts data as both a string and a h5py file. 
        """
        if type(data) is str:
            self.data = h5py.File(data, 'r')
        elif type(data) is h5py.File:
            self.data = data
        else:
            raise NotImplementedError(f"data of type {type(data)} is not supported.")

        self.data_key = data_key
        self.dtype = dtype
        self.indxs = indxs
        
        if len(args) > 0:
            for arg in args:
                assert len(arg) == len(self.indxs), \
                        "All given data must have the same length."

        self.args = list(args)

    def add(self, element):
        """ adds a new element to the dataset """
        assert len(element) == self.__len__(), \
                "All given data must have the same length."
        self.args.append(element)

    def __len__(self):
        return len(self.indxs)

    def __getitem__(self, indx):
        if torch.is_tensor(indx):
            indx = indx.tolist()
        
        indx_struct = self.indxs[indx]
        if torch.is_tensor(indx_struct):
            indx_struct = indx_struct.tolist()
        
        return torch.from_numpy( \
            self.data[self.data_key][indx_struct, 0, ...]).unsqueeze(0).type(self.dtype), \
            *[arg[indx] for arg in self.args]
