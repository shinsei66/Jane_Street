import pandas as pd
import numpy as np
import os
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.nn.modules.loss import _WeightedLoss, _Loss
import logging
import lightgbm as lgb
from lightgbm.callback import _format_eval_result
import pickle
from typing import Union, Optional, List
import collections


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

    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0).to(self.device)

    def forward(self, x):
        scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
        sampled_noise = self.noise.repeat(*x.size()).float().normal_() * scale
        x = x + sampled_noise
        return x

# https://github.com/L1aoXingyu/pytorch-beginner/blob/master/08-AutoEncoder/simple_autoencoder.py
# https://discuss.pytorch.org/t/pytorch-equivalent-of-keras/29412/2


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
        self.hidden = nn.Linear(input_size * 2, 640)
        self.bat = nn.BatchNorm1d(640)
        self.drop = nn.Dropout(dropout_rate)
        self.hidden2 = nn.Linear(640, output_size)
        self.act = nn.Sigmoid()
        self.x_feat = None
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(input_size),
            # GaussianNoise(sigma=noise),
            nn.Dropout(dropout_rate),
            nn.Linear(input_size, 640),
            # nn.SiLU()
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(640, input_size)
        )
        self.layer = nn.Sequential(
            nn.Linear(input_size * 2, 640),
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
        x = torch.cat((x, x_input), 1)
        x = self.layer(x)
        # x = self.act(x) #Use when classification
        return x

    def feature(self, x):
        x_input = x.clone()
        x = self.encoder(x)
        x = self.decoder(x)
        x = torch.cat((x, x_input), 1)
        return x


class SmoothBCEwLogits(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth(targets: torch.Tensor, n_labels: int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = targets * (1.0 - smoothing) + 0.5 * smoothing
        return targets

    def forward(self, inputs, targets):
        if len(inputs) > len(targets):
            inputs = inputs[:len(targets), :]
        elif len(inputs) < len(targets):
            targets = targets[:len(inputs), :]
        else:
            pass

        targets = SmoothBCEwLogits._smooth(targets, inputs.size(-1),
                                           self.smoothing)
        loss = F.binary_cross_entropy_with_logits(inputs, targets, self.weight)

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()

        return loss


# https://www.kaggle.com/a763337092/pytorch-resnet-starter-inference
class ResNetModel(nn.Module):
    def __init__(self, **kwargs):
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

        self.dense2 = nn.Linear(hidden_size + input_size, hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.dense3 = nn.Linear(hidden_size + hidden_size, hidden_size)
        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(dropout_rate)

        self.dense4 = nn.Linear(hidden_size + hidden_size, hidden_size)
        self.batch_norm4 = nn.BatchNorm1d(hidden_size)
        self.dropout4 = nn.Dropout(dropout_rate)

        self.dense5 = nn.Linear(hidden_size + hidden_size, output_size)

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
#         self.x_feat = torch.cat([x3, x4], 1).clone()

        x = self.dense5(x)
#         x = self.act(x)

        return x

    def feature(self, x):
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

        return x


class MLPNet (nn.Module):
    '''
    >> model =
        MLPNet(input_size = X.shape[-1], output_size = y.shape[-1] ).to(DEVICE)
    '''

    def __init__(self, **kwargs):
        super(MLPNet, self).__init__()
        input_size = kwargs['input_size']
        output_size = kwargs['output_size']
        self.dr = 0.2
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, output_size)
        self.dropout1 = nn.Dropout(self.dr)
        self.dropout2 = nn.Dropout(self.dr)
        self.bat = nn.BatchNorm1d(512)
        self.act = nn.Sigmoid()
        self.layer = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, 512),
            nn.SiLU(),
            nn.Linear(512, 512 * 2),
            nn.BatchNorm1d(512 * 2),
            nn.SiLU(),
            nn.Dropout(self.dr),
            nn.Linear(512 * 2, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(self.dr)
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
        self.device = device
        self.i = i
        ac_b = np.where(np.mean(self.output.cpu().detach(
        ).numpy(), axis=1) > 0, 1, 0).astype(int).copy()
        bat_size = len(ac_b)
        date = self.dat['date'][self.i * bat_size:(self.i + 1) * bat_size]
        weight = self.dat['weight'][self.i * bat_size:(self.i + 1) * bat_size]
        resp = self.dat['resp'][self.i * bat_size:(self.i + 1) * bat_size]
#         print('{}  {}  {}  {}'.format(date.shape, weight.shape, resp.shape, ac_b[:len(date)].shape))
        if (date.shape[0] == 0) or (ac_b.sum() == 0):
            self.us = 0
        else:
            self.us = utility_score_bincount(
                date, weight, resp, ac_b[:len(date)])

    def forward(self, input, target):

        #         loss = torch.tensor(-1*self.us, dtype=torch.float).to(self.device) * F.l1_loss(input, target)
        loss = torch.tensor(-1 * self.us, dtype=torch.float).to(
            self.device) * F.binary_cross_entropy(input, target)
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


class TransformerModel(nn.Module):
    '''
    >> model =
        TransformerModel(input_size = X.shape[-1], output_size = y.shape[-1], batch_size = BATCH_SIZE).to(DEVICE)
    '''

    def __init__(self, **kwargs):
        super(TransformerModel, self).__init__()
        self.input_size = kwargs['input_size']
        self.output_size = kwargs['output_size']
        self.batch_size = kwargs['batch_size']
        self.head = 8
        self.num_hidden = 512
        self.dr = 0.1
        self.dropout1 = nn.Dropout(self.dr)
#         self.conv1 = nn.Conv1d(in_channels=self.num_hidden, out_channels=self.num_hidden, kernel_size=1)
#         self.relu1 = nn.PReLU()

        self.layer0 = nn.Sequential(
            nn.BatchNorm1d(self.input_size),
            nn.Linear(self.input_size, self.num_hidden),
            Swish_module()
        )
        self.layer1 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.num_hidden,
                nhead=self.head,
                dropout=self.dr),
            num_layers=1)
        self.layer2 = nn.Linear(self.num_hidden, self.output_size)

    def forward(self, x):
        # input (batch size, input size)
        x = self.layer0(x).unsqueeze(2)
        # input (batch size, hidden size, 1)
        x = x.permute(2, 0, 1)
        # input (1, batch size, hidden size)
        x = self.layer1(x)
        # x (1, batch size, hidden size)
        x = x.squeeze()
        # x (batch size, hidden size))
        output = self.layer2(x)
        # x (batch size,output size))
        return output


def train_model(
        model,
        criterion,
        optimizer,
        scheduler,
        loaders,
        device,
        num_epoch,
        patiance,
        model_path,
        model_name,
        version,
        fold,
        logger,
        dat):
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
    es = Early_Stopping('min', patiance)
    counter = es.counter
    epoch_list, score_list, score_list_tr = [], [], []
    train_loader = loaders['train']
    valid_loader = loaders['valid']

    for epoch in tqdm(range(num_epoch)):
        score, tr_score = 0, 0
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
                for i, data in enumerate(train_loader):
                    optimizer.zero_grad()
                    x = data['x'].to(device)
                    y = data['y'].to(device)

                    # ===================forward=====================
                    output = model(x)
                    loss = criterion(output, y)
                    tr_score += loss.data.to('cpu').detach().numpy().copy() / \
                        len(train_loader)
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
                        score += loss.data.to('cpu').detach().numpy().copy() / \
                            len(valid_loader)

                    x_vl = torch.tensor(
                        dat['x_vl'], dtype=torch.float).to(device)
#                     pred = model(x_vl).cpu().detach().numpy()
                    pred = model(x_vl).sigmoid().cpu().detach().numpy()
                    action = np.where(np.mean(pred, axis=1) >
                                      0.5, 1, 0).astype(int).copy()
                    uscore = utility_score_bincount(date=dat['date'][:action.shape[0]],
                                                    weight=dat['weight'][:action.shape[0]],
                                                    resp=dat['resp'][:action.shape[0]],
                                                    action=action)
                    score = -1 * uscore
                    stop, counter, best_score,best_model, save_path\
                    = es(score, model, model_path, model_name, version, fold, epoch, logger)

                
                if type(scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                    scheduler.step(score)
                else:
                    scheduler.step()
        print(stop)
        if stop==True:
            logger.info('Loss did not improved for {} epochs'.format(patiance))
            break
        epoch_list.append(epoch + 1)
        score_list.append(score)
        score_list_tr.append(tr_score)

        logger.info(
            'Epoch [{}/{}],        Train  loss: {:.4f},         Valid loss: {:.4f},  \
        utility score: {:.4f},       Early stopping counter: {}' .format(
                epoch + 1, num_epoch, tr_score, score, uscore, counter))

    logger.info('The best loss is {:.6f}'.format(best_score))
    learn_hist = save_epoch_history(epoch_list, score_list_tr, score_list)
    model.load_state_dict(best_model)
    return model, learn_hist, save_path


# https://amalog.hateblo.jp/entry/lightgbm-logging-callback
def log_evaluation(logger, period=1, show_stdv=True, level=logging.DEBUG):
    def _callback(env):
        if period > 0 and env.evaluation_result_list and (
                env.iteration + 1) % period == 0:
            result = '\t'.join([_format_eval_result(x, show_stdv)
                                for x in env.evaluation_result_list])
            logger.log(level, '[{}]\t{}'.format(env.iteration + 1, result))
    _callback.order = 10
    return _callback


def train_lgb(
        param,
        data,
        num_round,
        disp,
        patiance,
        save,
        logger,
        save_path,
        n):
    """
    >> model = train_lgb(param, data, num_round, disp, PATIANCE, True, logger,  save_path, n)
    parameters
    ===========
    param:
    data:
    num_round:
    disp:
    patiance:
    save:
    logger:
    save_path:
    n: index to show which resp the lgb trained by resp_n


    returns
    ===========
    model: lightgbm predictor
    """

    trn_data, val_data = data['trn'], data['val']
    callbacks = [log_evaluation(logger, period=disp)]
    model = lgb.train(
        param,
        trn_data,
        num_round,
        valid_sets=[
            trn_data,
            val_data],
        verbose_eval=disp,
        early_stopping_rounds=patiance,
        callbacks=callbacks)

    if save:
        save_model(model, save_path, f'lightgbm_resp_{n}', 'pickle', logger)
    return model


def save_model(
    model:Union[lgb.basic.Booster, collections.OrderedDict], 
    save_path:str, 
    filename:str, 
    extension:str, 
    logger:Optional[logging.Logger]=None) -> None:
    """save model weight parameters
    >> save_model(model, save_path, filename, extension)
    
    Parameters:
    ===============
    model: model weight files
    save_path: the directory path of the model to be saved
    filename: the filename of the model weight files
    extension: extension type of the model files
    
    Returns:
    ===============
    None
    """

    allpath = f'{save_path}/{filename}.{extension}'
    if not os.path.exists(f'{save_path}/'):
            os.mkdir(f'{save_path}')
            
    if extension == 'pickle':
        with open(allpath, mode='wb') as f:
            pickle.dump(model, f)
        if logger:
            logger.info('Successfully saved in {}'.format(allpath))
    elif extension in ('pth'):
        torch.save(model, allpath)
        if logger:
            logger.info('Successfully saved in {}'.format(allpath))

def save_epoch_history(
    epoch_list:List, 
    train_loss_list:List, 
    valid_loss_list: List) -> pd.core.frame.DataFrame:
    """saving training loss histories
    >> learn_hist = save_epoch_history(epoch_list, train_loss_list, valid_loss_list)
    
    Parameters:
    ===============
    epoch_list: the list of the training epochs
    train_loss_list: the list of the train losses
    valid_loss_list: the list of the valid losses
    
    Returns:
    ===============
    learn_hist: pandas dataframe of the learning history
    """
    learn_hist = pd.DataFrame()
    learn_hist['epoch'] = epoch_list
    learn_hist['train_loss'] = train_loss_list
    learn_hist['valid_loss'] = valid_loss_list
    
    return learn_hist

class Early_Stopping():
    """early stopping function
    >> es = Early_Stopping('min', 1)
    
    Parameters:
    ===============
    mode: 'min' or 'max'
    patiance:
    
    Returns:
    ===============
   
    """
    
    def __init__(self, mode:str, patiance:int):
        self.mode = mode
        if mode == 'min':
            self.best_score = np.inf
        elif mode == 'max':
            self.best_score = -np.inf
        self.patiance = patiance
        self.counter = 0
        self.if_stop = False
        self.best_model = None
    
    
    def __call__(self, score, model, model_path, model_name, version, fold, epoch, logger) -> bool:
        """early stopping function
        >> stop, counter, best_score,best_model, save_path
        = Early_Stopping(score, model, model_path, model_name, version, fold, epoch, logger)

        Parameters:
        ===============
        score:
        model_path:
        model_name:
        version:
        fold:
        epoch:
        logger:

        Returns:
        ===============
        if_stop: boolean controller to stop training epochs
        """
        save_path= f'{model_path}/{model_name}_{version}'
        if np.round(score, decimals=5) < self.best_score:
            self.counter = 0
            self.best_score = score
            self.best_model = model.state_dict().copy()
            filename = f'{model_name}_fold_{fold}_' + str(epoch + 1)            
            save_model(self.best_model, save_path, filename, 'pth', logger)
            return self.if_stop, self.counter, self.best_score, self.best_model, save_path

        else:
            self.counter += 1
                        
            if self.counter == self.patiance:
                self.if_stop = True
                return self.if_stop, self.counter, self.best_score, self.best_model, save_path
            else:
                return self.if_stop, self.counter, self.best_score, self.best_model, save_path
