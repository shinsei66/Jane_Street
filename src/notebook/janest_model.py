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
from torch.nn.modules.loss import _WeightedLoss, _Loss
from sklearn.metrics import roc_auc_score, roc_curve, log_loss
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
        dropout_rate = 0.4
        self.hidden = nn.Linear(input_size*2, 640)
        self.bat = nn.BatchNorm1d(640)
        self.drop = nn.Dropout(dropout_rate)
        self.hidden2 = nn.Linear(640, output_size)
        self.act = nn.Sigmoid()
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(input_size),
            #GaussianNoise(sigma=noise),
            nn.Dropout(dropout_rate),
            nn.Linear(input_size, 640),
            #nn.SiLU()
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(640, input_size)
        )
        self.layer = nn.Sequential(
            nn.Linear(input_size*2, 640),
            nn.BatchNorm1d(640),
            nn.Dropout(dropout_rate),
            nn.Linear(640, 320),
            nn.Dropout(dropout_rate),
            nn.Linear(320, 640),
            nn.Linear(640, output_size)
        )

    def forward(self, x):
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
        self.dr = 0.2
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, output_size)
        self.dropout1 = nn.Dropout(self.dr )
        self.dropout2 = nn.Dropout(self.dr )
        self.bat = nn.BatchNorm1d(512)
        self.act = nn.Sigmoid()
        self.layer = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, 512),
            nn.SiLU(),
            nn.Linear(512, 512*2),
            nn.BatchNorm1d(512*2),
            nn.SiLU(),
            nn.Dropout(self.dr ),
            nn.Linear(512*2, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(self.dr )
        )
    def forward(self, x):
        x = self.layer(x)
        x = self.fc2(x)
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
    def __init__(self, dataset, target, date, weight, resp):
        self.dataset = dataset
        self.target = target
        self.weight = weight

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, item):
        return {
            'x': torch.tensor(self.dataset[item, :], dtype=torch.float),
            'y': torch.tensor(self.target[item, :], dtype=torch.float),
        }    

    


    
def utility_score_bincount(date, weight, resp, action):
    count_i = len(np.unique(date))
    # print('weight: ', weight)
    # print('resp: ', resp)
    # print('action: ', action)
    # print('weight * resp * action: ', weight * resp * action)
    Pi = np.bincount(date, weight * resp * action)
    t = np.sum(Pi) / np.sqrt(np.sum(Pi ** 2)) * np.sqrt(250 / count_i)
    u = np.clip(t, 0, 6) * np.sum(Pi)
    return u



    
class unity_loss(_Loss):
    def __init__(self, output, dat, device, i):
        super(unity_loss, self).__init__()
        self.output = output
        self.dat = dat
        self.device=device
        self.i = i
        ac_b = np.where(np.mean(self.output.cpu().detach().numpy(), axis=1)> 0, 1, 0).astype(int).copy()
        bat_size = len(ac_b)
        date = self.dat['date'][self.i*bat_size:(self.i+1)*bat_size]
        weight = self.dat['weight'][self.i*bat_size:(self.i+1)*bat_size]
        resp = self.dat['resp'][self.i*bat_size:(self.i+1)*bat_size]
#         print('{}  {}  {}  {}'.format(date.shape, weight.shape, resp.shape, ac_b[:len(date)].shape))
        if (date.shape[0] == 0 ) or( ac_b.sum() ==0): 
            self.us = 0
        else:
            self.us = utility_score_bincount(date, weight, resp, ac_b[:len(date)]) 

    def forward(self, input, target):

#         loss = torch.tensor(-1*self.us, dtype=torch.float).to(self.device) * F.l1_loss(input, target)
        loss = torch.tensor(-1*self.us, dtype=torch.float).to(self.device) * F.binary_cross_entropy(input, target)
        return loss

    
    
    
sigmoid = torch.nn.Sigmoid()
class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

swish = Swish.apply

class Swish_module(nn.Module):
    def forward(self, x):
        return swish(x)

class GRUModel(nn.Module):
    
    def __init__(self, *kwargs):
        '''
        org_feature denote the original features
        '''
        super(GRUModel, self).__init__()
        input_size = kwargs['input_size']
        output_size = kwargs['output_size']
        num_hidden = 256
        
        self.cnn1 = nn.Sequential(nn.Conv1d(input_size , output_size//2, kernel_size=1, padding=0, stride=1),
                                  Swish_module())
        self.cnn2 = nn.Sequential(nn.Conv1d(input_size , output_size//2, kernel_size=3, padding=1, stride=1),
                                  Swish_module())
        
        self.gru1     = nn.GRU(output_size, num_hidden, batch_first=True, bidirectional=True)
        self.dropout  = nn.Dropout(0.75)
        self.gru2     = nn.GRU((num_hidden*2+output_size), num_hidden, batch_first=True, bidirectional=True)
        self.gru3     = nn.GRU((num_hidden*4+output_size), num_hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(#nn.Dropout(0.5),
                                nn.Linear((num_hidden*6+output_size+output_size), num_hidden*2),
                                nn.ReLU(),
                                nn.Linear(num_hidden*2, output_size))
                
 
    def forward(self, input, mydata):
        
        input = torch.transpose(input, 1, 2)
        cnn_output1 = self.cnn1(input)
        cnn_output2 = self.cnn2(input)
        cnn_output  = torch.cat((cnn_output1, cnn_output2), 1)
        rnn_input   = torch.transpose(cnn_output, 1, 2)
        
        gru1, _     = self.gru1(rnn_input)
        gru1        = self.dropout(gru1)
        gru1        = torch.cat((rnn_input, gru1), 2) #densely connected recurrent network https://arxiv.org/pdf/1707.06130.pdf
        gru2, _     = self.gru2(gru1)
        gru2        = self.dropout(gru2)
        gru2        = torch.cat((gru1, gru2), 2)
        gru3, _     = self.gru3(gru2)
        gru3        = self.dropout(gru3)
        gru3        = torch.cat((gru2, gru3), 2)
        
        input_final = torch.cat((gru3, rnn_input),2)
        output = self.fc(input_final)
        
        return output 


    
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
                for i, data in enumerate(train_loader):
                    optimizer.zero_grad()
                    x = data['x'].to(device)
                    y = data['y'].to(device)
               
                    # ===================forward=====================
                    output = model(x)
#                     cr = unity_loss(output, dat, device, i)
#                     loss = cr(output, y)
                    loss = criterion(output, y)

                    tr_score +=  loss.data.to('cpu').detach().numpy().copy()/len(train_loader)
                    # ===================backward====================
                    loss.backward()
                    optimizer.step()

                    
            else:
                model.eval()
                with torch.no_grad():
                    for i, data in enumerate(valid_loader):
                        
                        x = data['x'].to(device)
                        y = data['y'].to(device)

                        output = model(x)
                        loss = torch.mean(criterion(output, y))
#                         cr = unity_loss(output, dat, device, i)
#                         loss += cr(output, y)
                        score +=  loss.data.to('cpu').detach().numpy().copy()/len(valid_loader)

                    x_vl = torch.tensor(dat['x_vl'], dtype=torch.float).to(device)
#                     pred = model(x_vl).cpu().detach().numpy()
                    pred = model(x_vl).sigmoid().cpu().detach().numpy()
                    action = np.where(np.mean(pred, axis=1)> 0.5, 1, 0).astype(int).copy()
                    uscore = utility_score_bincount(date=dat['date'] , weight= dat['weight'], resp=dat['resp'] , action = action)
                    score = -1*uscore
                    
                    if np.round(score,decimals=5) < best_score:
                        counter  = 0
                        best_score = score
                        best_model = model.state_dict().copy()
                        if not os.path.exists(f'{model_path}/{model_name}_{version}/'):
                            os.mkdir(f'{model_path}/{model_name}_{version}')
                        save_path = f'{model_path}/{model_name}_{version}/{model_name}_fold_{fold}_'+str(epoch+ 1)+'.pth'
                        
                    else:
                        counter += 1
                        
                plateau = True
                if plateau:
                    scheduler.step(score)
                else:
                    scheduler.step()

        if counter == patiance:
            logger.info('Loss did not improved for {} epochs'.format(patiance))
            break
        epoch_list.append(epoch+1)
        score_list.append(score)
        score_list_tr.append(tr_score)
        

        logger.info('Epoch [{}/{}],        Train  loss: {:.4f},         Valid loss: {:.4f},        utility score: {:.4f},       Early stopping counter: {}'\
              .format(epoch + 1, num_epoch, tr_score,  score, uscore, counter))
    torch.save(best_model, save_path)
    logger.info('The best loss is {:.6f}'.format(best_score))
    learn_hist = pd.DataFrame()
    learn_hist['epoch'] = epoch_list
    learn_hist['valid_loss'] = score_list
    learn_hist['train_loss'] = score_list_tr
    
    model.load_state_dict(best_model)
    
    return model, learn_hist, save_path


##WIP
def train_model_unity(model, criterion, optimizer, scheduler, loaders, device, num_epoch, patiance, \
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
                for i, data in enumerate(train_loader):
                    optimizer.zero_grad()
                    x = data['x'].to(device)
                    y = data['y'].to(device)
               
                    # ===================forward=====================
                    output = model(x)
                    cr = unity_loss(output, dat, device, i)
                    loss = cr(output, y)
#                     loss = criterion(output, y)
                    tr_score +=  loss.data.to('cpu').detach().numpy().copy()/len(train_loader)
                    # ===================backward====================
                    loss.backward()
                    optimizer.step()

                    
            else:
                model.eval()
                with torch.no_grad():
                    for i, data in enumerate(valid_loader):
                        
                        x = data['x'].to(device)
                        y = data['y'].to(device)

                        output = model(x)
#                         loss = torch.mean(criterion(output, y))
                        cr = unity_loss(output, dat, device, i)
                        loss = torch.mean(cr(output, y))
                        score +=  loss.data.to('cpu').detach().numpy().copy()/len(valid_loader)

                    x_vl = torch.tensor(dat['x_vl'], dtype=torch.float).to(device)
#                     pred = model(x_vl).cpu().detach().numpy()
                    pred = model(x_vl).sigmoid().cpu().detach().numpy()
                    action = np.where(np.mean(pred, axis=1)> 0.5, 1, 0).astype(int).copy()
                    uscore = utility_score_bincount(date=dat['date'] , weight= dat['weight'], resp=dat['resp'] , action = action)
                    score = -1*uscore
                    
                    if np.round(score,decimals=5) < best_score:
                        counter  = 0
                        best_score = score
                        best_model = model.state_dict().copy()
                        if not os.path.exists(f'{model_path}/{model_name}_{version}/'):
                            os.mkdir(f'{model_path}/{model_name}_{version}')
                        save_path = f'{model_path}/{model_name}_{version}/{model_name}_fold_{fold}_'+str(epoch+ 1)+'.pth'
                        
                    else:
                        counter += 1
                        
                plateau = True
                if plateau:
                    scheduler.step(score)
                else:
                    scheduler.step()

        if counter == patiance:
            logger.info('Loss did not improved for {} epochs'.format(patiance))
            break
        epoch_list.append(epoch+1)
        score_list.append(score)
        score_list_tr.append(tr_score)
        

        logger.info('Epoch [{}/{}],        Train  loss: {:.4f},         Valid loss: {:.4f},        utility score: {:.4f},       Early stopping counter: {}'\
              .format(epoch + 1, num_epoch, tr_score,  score, uscore, counter))
    torch.save(best_model, save_path)
    logger.info('The best loss is {:.6f}'.format(best_score))
    learn_hist = pd.DataFrame()
    learn_hist['epoch'] = epoch_list
    learn_hist['valid_loss'] = score_list
    learn_hist['train_loss'] = score_list_tr
    
    model.load_state_dict(best_model)
    
    return model, learn_hist, save_path