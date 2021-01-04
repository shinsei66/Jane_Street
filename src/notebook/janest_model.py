import pandas as pd
import numpy as np
import cupy as cp
import os
import gc
import time
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader
#print(torch.__version__)
#import matplotlib.pyplot as plt
from numba import njit
#%matplotlib inline




#https://github.com/L1aoXingyu/pytorch-beginner/blob/master/08-AutoEncoder/simple_autoencoder.py
#https://discuss.pytorch.org/t/pytorch-equivalent-of-keras/29412/2
class autoencoder(nn.Module):
    '''
    >> model = 
        autoencoder(input_size = X.shape[-1], output_size = y.shape[-1],\
        noise = 0.1).to(DEVICE)
    '''
    def __init__(self, **kwargs):
        super(autoencoder, self).__init__()
        input_size = kwargs['input_size']
        output_size = kwargs['output_size']
        noise = kwargs['noise']
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(input_size),
            #GaussianNoise(noise),
            nn.Linear(input_size, 640),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(640, input_size)
        )
        self.hidden = nn.Linear(input_size, 320)
        self.bat = nn.BatchNorm1d(320)
        self.drop = nn.Dropout(0.2)
        self.hidden2 = nn.Linear(320, output_size)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.hidden(x)
        x = self.bat(x)
        x = self.drop(x)
        x = self.hidden2(x)
        x = self.act(x)
        return x
    
    

    
    
class MLPNet (nn.Module):
    '''
    >> model = 
        MLPNet(input_size = X.shape[-1], output_size = y.shape[-1] ).to(DEVICE)
    '''
    def __init__(self,  **kwargs):
        super(MLPNet, self).__init__()
        input_size = kwargs['input_size']
        output_size = kwargs['output_size']
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, output_size)
        self.dropout1 = nn.Dropout2d(0.2)
        self.dropout2 = nn.Dropout2d(0.2)
        self.bat = nn.BatchNorm1d(512)
        self.act = nn.Sigmoid()
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, 512),
            nn.ReLU(True)
        )
        
    def forward(self, x):
        num_lyr = 5
        x = self.encoder(x)
        x = F.relu(self.bat(x))
        x = self.dropout1(x)
        for lyr in range(num_lyr):
            x = F.relu(self.fc1(x))
            x = self.dropout2(x)
        x = self.act(self.fc2(x))
        return x

    
class CustomDataset:
    def __init__(self, dataset, target):
        self.dataset = dataset
        self.target = target

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, item):
        return {
            'x': torch.tensor(self.dataset[item, :], dtype=torch.float),
            'y': torch.tensor(self.target[item, :], dtype=torch.float)
        }

    
    
    
    
def train_model(model, criterion, optimizer, loaders, device, num_epoch, patiance, \
                 model_path, model_name, version, fold):
    '''
    arguments
    ============
    model :
    criterion :
    optimizer :
    loaders :
    device :
    num_epoch :
    patiance : :
    model_path :
    model_name :
    version :
    fold :
    
    returns
    ============
    model : trained model's parameters
    learn_hist : learning curve data for each fold
    save_path : the best epoch model parameters file path for each fold
    '''
    best_score = 100
    counter = 0
    epoch_list = []
    score_list = []
    score_list_tr = []
    train_loader = loaders['train']
    valid_loader = loaders['valid']
    for epoch in tqdm(range(num_epoch)):
        score = 0
        tr_score = 0
        for phase in ['train', 'valid']:
            if phase == 'train':
                
                for data in train_loader:
                    x = data['x'].to(device)
                    y = data['y'].to(device)
                    model.train()
                    # ===================forward=====================
                    output = model(x)
                    loss = criterion(output, y)
                    tr_score +=  loss.data.to('cpu').detach().numpy().copy()
                    # ===================backward====================
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
            else:
                with torch.no_grad():
                    for data in valid_loader:
                        x = data['x'].to(device)
                        y = data['y'].to(device)
                        model.eval()
                        output = model(x)
                        loss = criterion(output, y)
                        score +=  loss.data.to('cpu').detach().numpy().copy()

                    if score <= best_score:
                        counter  = 0
                        best_score = score.copy()
                        if not os.path.exists(f'{model_path}/{model_name}_{version}/'):
                            os.mkdir(f'{model_path}/{model_name}_{version}')
                        save_path = f'{model_path}/{model_name}_{version}/{model_name}_fold_{fold}_'+str(epoch+ 1)+'.pth'
                        best_model = model.state_dict()

                    else:
                        counter += 1
        if counter == patiance:
            print('Loss did not improved for {} epochs'.format(patiance))
            torch.save(best_model, save_path)
            print('The best bse loss is {:.4f}'.format(best_score))
            break
        epoch_list.append(epoch+1)
        score_list.append(score)
        score_list_tr.append(tr_score)

        print('Epoch [{}/{}],        Train BCE loss: {:.4f},         Valid BCE loss: {:.4f},       Early stopping counter: {}'\
              .format(epoch + 1, num_epoch, tr_score,  score, counter))
    learn_hist = pd.DataFrame()
    learn_hist['epoch'] = epoch_list
    learn_hist['valid_bce_loss'] = score_list
    learn_hist['train_bce_loss'] = score_list_tr
    
    return model, learn_hist, save_path