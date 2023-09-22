#!/home/rusack/joshib/.conda/envs/fair_gpu/bin/python
import os,sys
from tqdm import tqdm
import pickle

import torch
from models.DNN import NN
from dataset_utils import H5DatasetDNN, split_dataset
from torch.utils.data.dataloader import default_collate
from utils.torch_utils import MARELoss, train

EPOCHS = 100
TRAIN_BATCH_SIZE = 10000
LAYERS = [50, 40, 30, 1] # shape of DNN
LRATE = 1e-2 # learning rate
TRAINING_FOLDER="../training/test_epochs_100_lr_0p01_bs_1e5"

if not os.path.exists(TRAINING_FOLDER):
        os.system('mkdir -p {}'.format(TRAINING_FOLDER))

file_path = '/home/rusack/shared/hdf5/hgcal_pion/hgcal_pions_combinedHgc_Ahc_1.h5'
dataset = H5DatasetDNN(file_path)
train_test_datasets = split_dataset(dataset)


# check if gpus are available
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda:0'

X = train_test_datasets['train']
Y = train_test_datasets['test']

dataloaders = { 'train': torch.utils.data.DataLoader(X, TRAIN_BATCH_SIZE, shuffle=False,
                         collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x))),
                'test': torch.utils.data.DataLoader(Y, len(Y), shuffle=False,
                        collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)))}

print("Training sample size: ", len(X), "Tseting sample size: ", len(Y))
nn = NN(LAYERS)
nn.to(device)
optimizer = torch.optim.Adam(nn.parameters(), lr = LRATE)
#optimizer = torch.optim.SGD(nn.parameters(), lr = LRATE)
loss_func = torch.nn.MSELoss()

epochs = []
lr_array = []
train_loss_array = []
valid_loss_array = []

pbar = tqdm(range(EPOCHS))
for epoch in pbar:
    train_loss = 0
    test_loss = 0
    nxtrain = 0
    for xtrain, ytrain in dataloaders['train']:
        tmptl, tmpto = train(nn, xtrain, ytrain, optimizer, loss_func)
        train_loss += tmptl.item()
        nxtrain += len(xtrain)
        
    train_loss = train_loss/nxtrain
        
    with torch.no_grad():
        for xtest, ytest in dataloaders['test']:
            xtest = nn(xtest).reshape(-1)
            tmpvl = loss_func(xtest, ytest)
            test_loss += tmpvl.item()
        test_loss = test_loss/len(ytrain)
            
    epochs.append(epoch)
    train_loss_array.append(train_loss)
    valid_loss_array.append(test_loss)
    lr_array.append(optimizer.param_groups[0]['lr'])

    pbar.set_postfix({'training loss': train_loss, 'validation loss': test_loss})
    torch.save(nn.state_dict(), f'{TRAINING_FOLDER}/epoch{epoch}')

training_summary = {
    'epochs': epochs,
    'train_loss': train_loss_array,
    'valid_loss': valid_loss_array,
    'learning_rate': lr_array
}

with open('{}/summary.pkl'.format(TRAINING_FOLDER),'wb') as f_:
    pickle.dump(training_summary, f_)
