import pandas as pd
import numpy as np
import glob
#import cupy as cp
import os
import gc
import time
import torch
import torchvision
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import nn
import torch.nn.functional as F
#from tqdm.notebook import tqdm
from torch.utils.data import DataLoader
print(torch.__version__)
#import matplotlib.pyplot as plt
from numba import njit
#%matplotlib inline
from janest_model import MLPNet , CustomDataset, train_model
from utils import PurgedGroupTimeSeriesSplit, utility_score_numba


TRAINING = True
USE_FINETUNE = True     
FOLDS = 5
GROUP_GAP = 20
SEED = 66
INPUTPATH = '../../input'
NUM_EPOCH = 500
BATCH_SIZE = 16384
PATIANCE = 15
LR = 0.0005
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
MDL_PATH  = '../models'
MDL_NAME = 'mlp'
NUM_LYR = 5
VER = 'cv_base_swish'


def main():
    f_mean = np.load( f'{INPUTPATH}/f_mean.npy')
    X = np.load( f'{INPUTPATH}/X.npy')
    y = np.load( f'{INPUTPATH}/y.npy')
    
    
    gkf =  PurgedGroupTimeSeriesSplit(n_splits = FOLDS,  group_gap = GROUP_GAP)
    model = MLPNet(input_size = X.shape[-1], output_size = y.shape[-1]).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'min',verbose=True,patience=5)
    print(model)
    
    
    
    sts = time.time()
    learn_hist_list = []
    save_path_list = []
    for fold, (tr, vl) in enumerate(gkf.split(train['action'].values, train['action'].values, train['date'].values)):
        print('Fold : {}'.format(fold+1))

        X_tr, X_val = X[tr], X[vl]
        y_tr, y_val = y[tr], y[vl]
        trn_dat = CustomDataset(X_tr, y_tr)
        val_dat = CustomDataset(X_val, y_val)
        trn_loader = DataLoader(trn_dat , batch_size=BATCH_SIZE, shuffle=False)
        val_loader = DataLoader(val_dat , batch_size=BATCH_SIZE, shuffle=False)
        loaders = {'train':trn_loader, 'valid': val_loader}
        trained_model, learn_hist, save_path =\
            train_model(model, criterion, optimizer, scheduler, loaders, DEVICE, NUM_EPOCH, PATIANCE, \
                    MDL_PATH, MDL_NAME, VER, fold+1)
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        plt.plot(learn_hist.epoch, learn_hist.valid_bce_loss, color = 'blue')
        ax2 = ax1.twinx()
        plt.plot(learn_hist.epoch, learn_hist.train_bce_loss, color = 'red')
        ax1.set_ylabel('Valid BCE Loss')
        ax2.set_ylabel('Train BCE Loss')
        plt.xlabel('Epochs')
        plt.title('Learning Curve')
        fig_path = f'{MDL_PATH}/{MDL_NAME}_{VER}/figures'
        if not os.path.exists(fig_path):
                            os.mkdir(fig_path)
        plt.savefig(fig_path+f'learning_hist_fold{fold}.png')
        learn_hist['Fold'] = fold+1
        learn_hist_list.append(learn_hist)
        save_path_list.append(save_path)
    hist_path = f'{MDL_PATH}/{MDL_NAME}_{VER}/history'
    if not os.path.exists(hist_path):
                        os.mkdir(hist_path)
    all_hist = pd.concat(learn_hist_list, axis=0)
    all_hist.reset_index(inplace=True, drop=True)
    all_hist.to_csv(f'{MDL_PATH}/{MDL_NAME}_{VER}/{MDL_NAME}_learning_history.csv', index=False)
    ed = time.time()
    print('Training process takes {:.2f} min.'.format((ed-sts)/60))


if __name__ == "__main__":
    main()