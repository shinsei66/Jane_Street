import warnings
warnings.filterwarnings('ignore')

import os, gc, random, time, sys
# import cudf
import pandas as pd
import numpy as np

# from hyperopt import hp, fmin, tpe, Trials
# from hyperopt.pyll.base import scope
from sklearn.metrics import roc_auc_score, roc_curve, log_loss
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from joblib import dump, load

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

import yaml
import argparse
import logging
from utils import get_args

args = get_args()
with open(args.config_path, 'r') as f:
    config = yaml.safe_load(f)


EXT = config['EXT']
TRAINING = config['TRAINING']
USE_FINETUNE = config['USE_FINETUNE']     
FOLDS = config['FOLDS']
GROUP_GAP = config['GROUP_GAP']
SEED = config['SEED']
INPUTPATH = config['INPUTPATH']
NUM_EPOCH = config['NUM_EPOCH']
BATCH_SIZE = config['BATCH_SIZE']
PATIANCE = config['PATIANCE']
LR =config['LR']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
MDL_PATH  =config['MDL_PATH']
MDL_NAME =config['MDL_NAME']
VER = config['VER']
THRESHOLD = config['THRESHOLD']

format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level = logging.INFO,format=format_str, filename=f'../logs/{MDL_NAME}_{VER}_{EXT}.log')
logger = logging.getLogger('Log')
logger.info(config)
logger.info(sys.argv)





def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(seed=42)


print('Loading...')
train = pd.read_parquet(f'{INPUTPATH}/train.parquet')
features = [c for c in train.columns if 'feature' in c]

print('Filling...')
f_mean = train[features[1:]].mean()
train = train.loc[train.weight > 0].reset_index(drop = True)
train[features[1:]] = train[features[1:]].fillna(f_mean)
train['action'] = (train['resp'] > 0).astype('int')

print('Converting...')
# train = train.to_pandas()
f_mean = f_mean.values#.get()
# np.save('f_mean.npy', f_mean)

print('Finish.')

all_feat_cols = features #[col for col in feat_cols]
target_cols = ['action']


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.batch_norm0 = nn.BatchNorm1d(len(all_feat_cols))
        self.dropout0 = nn.Dropout(0.2)

        dropout_rate = 0.2
        hidden_size = 256
        self.dense1 = nn.Linear(len(all_feat_cols), hidden_size)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.dense2 = nn.Linear(hidden_size+len(all_feat_cols), hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.dense3 = nn.Linear(hidden_size+hidden_size, hidden_size)
        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(dropout_rate)

        self.dense4 = nn.Linear(hidden_size+hidden_size, hidden_size)
        self.batch_norm4 = nn.BatchNorm1d(hidden_size)
        self.dropout4 = nn.Dropout(dropout_rate)

        self.dense5 = nn.Linear(hidden_size+hidden_size, len(target_cols))

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

        return x

class MarketDataset:
    def __init__(self, df):
        self.features = df[features].values

        self.label = (df['resp'] > 0).astype('int').values.reshape(-1, 1)


    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return {
            'features': torch.tensor(self.features[idx], dtype=torch.float),
            'label': torch.tensor(self.label[idx], dtype=torch.float)
        }
    
    
def train_fn(model, optimizer, scheduler, loss_fn, dataloader, device):
    model.train()
    final_loss = 0

    for data in dataloader:
        optimizer.zero_grad()
        features = data['features'].to(device)
        label = data['label'].to(device)
        outputs = model(features)
        loss = loss_fn(outputs, label)
        loss.backward()
        optimizer.step()
        
            

        final_loss += loss.item()

    final_loss /= len(dataloader)
    if scheduler:
        plateau = True
        if plateau:
            scheduler.step(final_loss )
        else:
            scheduler.step()

    return final_loss

def inference_fn(model, dataloader, device):
    model.eval()
    preds = []

    for data in dataloader:
        features = data['features'].to(device)

        with torch.no_grad():
            outputs = model(features)

        preds.append(outputs.sigmoid().detach().cpu().numpy())

    preds = np.concatenate(preds).reshape(-1)

    return preds


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

class EarlyStopping:
    def __init__(self, patience=PATIANCE, mode="max", delta=0.):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def __call__(self, epoch_score, model, model_path):

        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
        elif score < self.best_score: #  + self.delta
            self.counter += 1
            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # ema.apply_shadow()
            self.save_checkpoint(epoch_score, model, model_path)
            # ema.restore()
            self.counter = 0

    def save_checkpoint(self, epoch_score, model, model_path):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            print('Validation score improved ({} --> {}). Saving model!'.format(self.val_score, epoch_score))
            
            torch.save(model.state_dict(), model_path)
        self.val_score = epoch_score

        
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


batch_size = BATCH_SIZE
label_smoothing = 1e-2
learning_rate = LR

start_time = time.time()
oof = np.zeros(len(train['action']))
gkf = GroupKFold(n_splits = FOLDS)
for fold, (tr, te) in enumerate(gkf.split(train['action'].values, train['action'].values, train['date'].values)):
    train_set = MarketDataset(train.loc[tr])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_set = MarketDataset(train.loc[te])
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=4)
    
    torch.cuda.empty_cache()
    device = DEVICE
    model = Model()
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = SmoothBCEwLogits(smoothing=label_smoothing)
    scheduler = ReduceLROnPlateau(optimizer, 'min',0.5,verbose=True,patience=5)
    
    ckp_path = f'{MDL_PATH }/{MDL_NAME}_{VER}/JSModel_{fold}.pth'
    model_path = f'{MDL_PATH }/{MDL_NAME}_{VER}/'
    if not os.path.exists(f'{model_path}'):
        os.mkdir(f'{model_path}')
    es = EarlyStopping(patience=PATIANCE, mode="max")
    for epoch in range(NUM_EPOCH ):
        train_loss = train_fn(model, optimizer, scheduler, loss_fn, train_loader, device)
        valid_pred = inference_fn(model, valid_loader, device)
        auc_score = roc_auc_score((train.loc[te]['resp'] > 0).astype('int').values.reshape(-1, 1), valid_pred)
        logloss_score = log_loss((train.loc[te]['resp'] > 0).astype('int').values.reshape(-1, 1), valid_pred)
        valid_pred = np.where(valid_pred >= 0.5, 1, 0).astype(int)
        u_score = utility_score_bincount(date=train.loc[te].date.values, weight=train.loc[te].weight.values, resp=train.loc[te].resp.values, action=valid_pred)

        logger.info(f"FOLD{fold} EPOCH:{epoch:3}, train_loss:{train_loss:.5f}, u_score:{u_score:.5f}, auc:{auc_score:.5f}, logloss:{logloss_score:.5f}, "
              f"time: {(time.time() - start_time) / 60:.2f}min")
        
        es(u_score, model, model_path=ckp_path)
        if es.early_stop:
            logger.info("Early stopping")
            break