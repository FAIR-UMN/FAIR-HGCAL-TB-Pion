import os,sys
from tqdm import tqdm
import pickle
import numpy as np

import torch

from models.DNN import NN
from dataset_utils import H5DatasetDNN, split_dataset
from torch.utils.data.dataloader import default_collate
from utils.torch_utils import MARELoss, train

EPOCHS = 100
TRAIN_BATCH_SIZE = 1
LAYERS = [50, 40, 30, 1] # shape of DNN
LRATE = 1e-2 # learning rate
file_path = '/home/rusack/shared/hdf5/hgcal_pion/hgcal_pions_combinedHgc_Ahc_2.h5'
# check if gpus are available
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda:0'

dataset = H5DatasetDNN(file_path)
TRAIN_BATCH_SIZE = len(dataset)
dataloaders = { 'full': torch.utils.data.DataLoader(dataset, TRAIN_BATCH_SIZE, shuffle=False,
                         collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x))) }
nn = NN(LAYERS)
nn.to(device)
#optimizer = torch.optim.Adam(nn.parameters(), lr = LRATE)
optimizer = torch.optim.SGD(nn.parameters(), lr = LRATE)
loss_func = torch.nn.MSELoss()
nn.load_state_dict(torch.load('../training/test_epochs_100_lr_0p01_bs_1e5/epoch99', map_location=torch.device(device)))
nn_output = None
with torch.no_grad():
    for xtest, ytest in dataloaders['full']:
        nn_output = nn(xtest).reshape(-1,)

with open('nn_output.pkl','wb') as f_:
    pickle.dump(nn_output, f_)
