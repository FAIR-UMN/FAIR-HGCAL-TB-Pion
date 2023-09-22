# Data Loader for DNN
import os
import h5py
import pickle
import numpy as np
import awkward as ak

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, Subset, DataLoader

# set the seed
np.random.seed(2137)
torch.manual_seed(2137)

HGCAL_X_Min = -36
HGCAL_X_Max = 36

HGCAL_Y_Min = -36
HGCAL_Y_Max = 36

HGCAL_Z_Min = 13
HGCAL_Z_Max = 265

HGCAL_E_Min = 0
HGCAL_E_Max = 2727

NHITS_Max = 2000

def split_dataset(dataset, split=0.8):
    size = len(dataset)
    train_size = int(split*size)
    train_idx, test_idx = train_test_split(list(range(size)), train_size=train_size)
    D = {
        'train': Subset(dataset, train_idx),
        'test': Subset(dataset, test_idx)
    }
    return D

class DNNDataset(Dataset):

    def __init__(self, filepath):
        if not os.path.exists(filepath):
            print('FIle does not exist!')
        
        self.nevents = 0
        self.input_size = 28
        self.dataset_size = 0
        self.data = None
        self.targets = None
        self.prepare()
    
    def normalize(self, data, var='xarray'):
        '''
        function to normalize variables
        '''
        limits = {
            'xarray': [HGCAL_X_Min, HGCAL_X_Max],
            'yarray': [HGCAL_X_Min, HGCAL_X_Max],
            'zarray': [HGCAL_Z_Min, HGCAL_Z_Max],
            'energy': [HGCAL_E_Min, HGCAL_E_Max]
        }
        
        return (data-limits[var][0])/(limits[var][1]-limits[var][0])
    
    def set_input_size(self, input_):
        self.input_size = input_
    
    def __len__(self):
        return self.nevents
    
    def __getitem__(self, index=1):
        return self.data[index], self.targets[index]
    
    def get_torched_element(self, x):
        x = ak.to_numpy(x).astype(np.float32)
        return Data(x=torch.from_numpy(x))
    
    def prepare(self):
        """
        Prepapre dataset
        """
        return

class H5DatasetDNN(DNNDataset):

    def __init__(self, h5_path):
        if os.path.exists(h5_path):
            self.h5_file = h5py.File(h5_path, 'r')
        else:
            print('File does not exist!')
        
        self.dataset_size = 0
        self.data = None
        self.targets = None
        self.prepare()
    
    def prepare(self):
        
        nhits = np.asarray(self.h5_file['nhits'], dtype=int)
        self.targets = np.asfarray(self.h5_file['true_energy'])
        self.nevents = len(nhits)

        zarray = np.asarray(self.h5_file['rechit_z'])
        
        # Use first 40 layers and then take every 4th layer instead
        all_layers = np.unique(zarray)
        layers = all_layers[0:40]
        for i in range(1, 11):
            layers = np.append(layers, all_layers[37+4*i])
        
        # retrieve rechit information
        zarray = ak.unflatten(counts=nhits, array=zarray)
        energy = ak.unflatten(counts=nhits, array=np.asarray(self.h5_file['rechit_energy']))

        self.normalize(zarray, var='zarray')
        self.normalize(energy, var='energy')

        # sum energies of all rechits in a layer
        self.data = np.array([ ak.sum(energy[zarray==l], axis=-1) for l in layers ], dtype=np.float32).T
        self.data = self.data/NHITS_Max
    

class PklDatasetDNN(DNNDataset):
    
    def __init__(self, pickle_path):

        self.pickle_list = ['rechit_z', 'rechit_energy', 'target']
        self.arrays = {}
        self.dataset_size = 0
        self.data = None
        self.targets = None

        if os.path.exists(pickle_path):
            for pkl in self.pickle_list:
                with open('%s/%s.pickle' % (pickle_path, pkl),'rb') as fpkl_:
                    self.arrays[pkl] = pickle.load(fpkl_)
        else:
            print('Files do not exist!')
        
        self.prepare()
          
    def prepare(self):
        
        self.targets = np.asarray(self.arrays['true_energy'])
        self.nevents = len(self.targets)

        zarray = self.arrays['rechit_z']
        energy = self.arrays['rechit_energy']
        layers = np.unique(np.asarray(ak.flatten(zarray)))[:self.input_size]

        self.normalize(zarray, var='zarray')
        self.normalize(energy, var='energy')

        # sum energies of all rechits in a layer
        self.data = np.array([ ak.sum(energy[zarray==l], axis=-1) for l in layers ], dtype=np.float32).T
        self.data = self.data/NHITS_Max
