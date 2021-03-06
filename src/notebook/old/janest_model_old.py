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
from tqdm import tqdm
#from tqdm.notebook import tqdm
from torch.utils.data import DataLoader
#print(torch.__version__)
import matplotlib.pyplot as plt
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn.modules.loss import _WeightedLoss
import logging


# https://discuss.pytorch.org/t/writing-a-simple-gaussian-noise-layer-in-pytorch/4694
class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self,  sigma=0.1, is_relative_detach=True ):
        super().__init__()
        self.sigma = sigma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0).to(self.device)

    def forward(self, x):
        scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
        sampled_noise = self.noise.repeat(*x.size()).float().normal_() * scale
        x = x + sampled_noise
        return x

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
#         device = kwargs['device']
        noise = kwargs['noise']
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(input_size),
            GaussianNoise(sigma=noise),
            nn.Linear(input_size, 640),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(640, input_size)
        )
        self.hidden = nn.Linear(input_size*2, 640)
        self.bat = nn.BatchNorm1d(640)
        self.drop = nn.Dropout(0.2)
        self.hidden2 = nn.Linear(640, output_size)
        self.act = nn.Sigmoid()
       

    def forward(self, x):
        x_input = x.clone()
        x = self.encoder(x)
        x = self.decoder(x)
        x = torch.cat((x, x_input),1)
        x = self.hidden(x)
        x = self.bat(x)
        x = self.drop(x)
        x = self.hidden2(x)
        #x = self.act(x) #Use when classification
        return x

    
class autoencoder2(nn.Module):
    '''
    >> model = 
        autoencoder(input_size = X.shape[-1], output_size = y.shape[-1],\
        noise = 0.1).to(DEVICE)
    '''
    def __init__(self, **kwargs):
        super(autoencoder2, self).__init__()
        input_size = kwargs['input_size']
        output_size = kwargs['output_size']
#         device = kwargs['device']
        noise = kwargs['noise']
        self.hidden = nn.Linear(input_size*2, 640)
        self.bat = nn.BatchNorm1d(640)
        #self.bat2 = nn.BatchNorm1d(input_size )
        self.drop = nn.Dropout(0.2)
        self.hidden2 = nn.Linear(640, output_size)
        self.act = nn.Sigmoid()
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(input_size),
            #GaussianNoise(sigma=noise),
            nn.Linear(input_size, 640),
            #nn.SiLU()
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(640, input_size)
        )
        self.layer = nn.Sequential(
            nn.Linear(input_size*2, 640),
            nn.BatchNorm1d(640),
            nn.Dropout(0.2),
            nn.Linear(640, 320),
            nn.Dropout(0.2),
            nn.Linear(320, 640),
            nn.Linear(640, output_size)
        )
            
       

    def forward(self, x):
        #x_input = self.bat2(x.clone())
        x_input = x.clone()
        x = self.encoder(x)
        x = self.decoder(x)
        x = torch.cat((x, x_input),1)
        x = self.layer(x)
        #x = self.act(x) #Use when classification
        return x

class SmoothBCEwLogits(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth(targets:torch.Tensor, n_labels:int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = targets * (1.0 - smoothing) + 0.5 * smoothing
        return targets

    def forward(self, inputs, targets):
        targets = SmoothBCEwLogits._smooth(targets, inputs.size(-1),
            self.smoothing)
        loss = F.binary_cross_entropy_with_logits(inputs, targets,self.weight)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()

        return loss
    
    
# https://www.kaggle.com/a763337092/pytorch-resnet-starter-inference
class ResNetModel(nn.Module):
    def __init__(self,**kwargs):
        super(ResNetModel, self).__init__()
        input_size = kwargs['input_size']
        output_size = kwargs['output_size']
        self.batch_norm0 = nn.BatchNorm1d(input_size)
        self.dropout0 = nn.Dropout(0.2)

        dropout_rate = 0.2
        hidden_size = 256
        self.act = nn.Sigmoid()
        self.dense1 = nn.Linear(input_size, hidden_size)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.dense2 = nn.Linear(hidden_size+input_size, hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.dense3 = nn.Linear(hidden_size+hidden_size, hidden_size)
        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(dropout_rate)

        self.dense4 = nn.Linear(hidden_size+hidden_size, hidden_size)
        self.batch_norm4 = nn.BatchNorm1d(hidden_size)
        self.dropout4 = nn.Dropout(dropout_rate)

        self.dense5 = nn.Linear(hidden_size+hidden_size, output_size)

        self.Relu = nn.ReLU(inplace=True)
        self.PReLU = nn.PReLU()
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        # self.GeLU = nn.GELU()
        self.RReLU = nn.RReLU()

    def forward(self, x):
        x = self.batch_norm0(x)
        x = self.dropout0(x)

        x1 = self.dense1(x)
        x1 = self.batch_norm1(x1)
        # x = F.relu(x)
        # x = self.PReLU(x)
        x1 = self.LeakyReLU(x1)
        x1 = self.dropout1(x1)

        x = torch.cat([x, x1], 1)

        x2 = self.dense2(x)
        x2 = self.batch_norm2(x2)
        # x = F.relu(x)
        # x = self.PReLU(x)
        x2 = self.LeakyReLU(x2)
        x2 = self.dropout2(x2)

        x = torch.cat([x1, x2], 1)

        x3 = self.dense3(x)
        x3 = self.batch_norm3(x3)
        # x = F.relu(x)
        # x = self.PReLU(x)
        x3 = self.LeakyReLU(x3)
        x3 = self.dropout3(x3)

        x = torch.cat([x2, x3], 1)

        x4 = self.dense4(x)
        x4 = self.batch_norm4(x4)
        # x = F.relu(x)
        # x = self.PReLU(x)
        x4 = self.LeakyReLU(x4)
        x4 = self.dropout4(x4)

        x = torch.cat([x3, x4], 1)

        x = self.dense5(x)
#         x = self.act(x) 

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
        self.dr = 0.5
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, output_size)
        self.dropout1 = nn.Dropout(self.dr )
        self.dropout2 = nn.Dropout(self.dr )
        self.bat = nn.BatchNorm1d(512)
        self.act = nn.Sigmoid()
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, 512),
            nn.ReLU(True)
        )
        self.layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(self.dr ),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(self.dr )
        )
    def forward(self, x):
        x = self.encoder(x)
        x = F.relu(self.bat(x))
        x = self.dropout1(x)
        x = self.layer(x)
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

class CustomDataset2:
    def __init__(self, dataset, target, weight):
        self.dataset = dataset
        self.target = target
        self.weight = weight

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, item):
        return {
            'x': torch.tensor(self.dataset[item, :], dtype=torch.float),
            'y': torch.tensor(self.target[item, :], dtype=torch.float),
            'w': torch.tensor(self.weight[item], dtype=torch.float)
        }    

    
    

    
    
    
def train_model(model, criterion, optimizer, scheduler, loaders, device, num_epoch, patiance, \
                 model_path, model_name, version, fold, logger, dat):
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
    logger :
    data:
    
    returns
    ============
    model : trained model's parameters
    learn_hist : learning curve data for each fold
    save_path : the best epoch model parameters file path for each fold
    '''
    from numba import njit
    @njit(fastmath = True)
    def utility_score_numba(date, weight, resp, action):
        Pi = np.bincount(date, weight * resp * action)
        t = np.sum(Pi) / np.sqrt(np.sum(Pi ** 2)) * np.sqrt(250 / len(Pi))
        u = min(max(t, 0), 6) * np.sum(Pi)
        return u
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
                model.train()
                for data in train_loader:
                    optimizer.zero_grad()
                    x = data['x'].to(device)
                    y = data['y'].to(device)
               
                    # ===================forward=====================
                    output = model(x)
                    loss = criterion(output, y)
                    tr_score +=  loss.data.to('cpu').detach().numpy().copy()/len(train_loader)
                    # ===================backward====================
                    loss.backward()
                    optimizer.step()
                    
            else:
                model.eval()
                with torch.no_grad():
                    for data in valid_loader:
                        x = data['x'].to(device)
                        y = data['y'].to(device)
                        
                        output = model(x)
                        loss = torch.mean(criterion(output, y))
                        score +=  loss.data.to('cpu').detach().numpy().copy()/len(valid_loader) 
                
                    if np.round(score,decimals=13) < best_score:
                        counter  = 0
                        best_score = score.copy()
                        if not os.path.exists(f'{model_path}/{model_name}_{version}/'):
                            os.mkdir(f'{model_path}/{model_name}_{version}')
                        save_path = f'{model_path}/{model_name}_{version}/{model_name}_fold_{fold}_'+str(epoch+ 1)+'.pth'
                        best_model = model.state_dict()

                    else:
                        counter += 1
                plateau = True
                if plateau:
                    scheduler.step(score)
                else:
                    scheduler.step()
        if counter == patiance:
            logger.info('Loss did not improved for {} epochs'.format(patiance))
            #torch.save(best_model, save_path)
            #print('The best bse loss is {:.6f}'.format(best_score))
            break
        epoch_list.append(epoch+1)
        score_list.append(score)
        score_list_tr.append(tr_score)
        model.load_state_dict(best_model)
        model.eval()
        x_vl = torch.tensor(dat['x_vl'], dtype=torch.float).to(device)
        pred = model(x_vl).cpu().detach().numpy()
        action = np.where(np.mean(pred, axis=1)> 0, 1, 0).astype(int).copy()
        uscore = utility_score_numba(dat['date'] , dat['weight'], dat['resp'] , action)

        logger.info('Epoch [{}/{}],        Train  loss: {:.6f},         Valid loss: {:.6f},        utility score: {:.6f},       Early stopping counter: {}'\
              .format(epoch + 1, num_epoch, tr_score,  score, uscore, counter))
    torch.save(best_model, save_path)
    logger.info('The best bse loss is {:.6f}'.format(best_score))
    learn_hist = pd.DataFrame()
    learn_hist['epoch'] = epoch_list
    learn_hist['valid_loss'] = score_list
    learn_hist['train_loss'] = score_list_tr
    
    return model, learn_hist, save_path