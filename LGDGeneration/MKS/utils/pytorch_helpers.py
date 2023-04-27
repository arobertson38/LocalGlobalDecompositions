"""
Hello!
This file contains useful code for training torch models. Code such as
custom data loaders. 

Created by: Andreas E. Robertson
Contact: arobertson38@gatech.edu
"""
import torch
import h5py

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
