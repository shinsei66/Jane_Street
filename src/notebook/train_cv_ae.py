import glob
#import cupy as cp
import os
import gc
import sys
import time
import yaml
import argparse
import logging
import pandas as pd
import numpy as np
import torch
import torchvision
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import nn
import torch.nn.functional as F
#from tqdm.notebook import tqdm
from tqdm import tqdm
from torch.utils.data import DataLoader
print(torch.__version__)
import matplotlib.pyplot as plt
from numba import njit
#%matplotlib inline
from janest_model import MLPNet , CustomDataset, train_model, autoencoder
from utils import PurgedGroupTimeSeriesSplit, get_args



def main():
    
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
    
    
    logging.basicConfig(level = 'INFO', filename=f'../logs/{MDL_NAME}_{VER}_{EXT}.log')
    logger = logging.getLogger('Log')
    logger.info(config)
    logger.info(sys.argv)
    
    f_mean = np.load( f'{INPUTPATH}/f_mean.npy')
    X = np.load( f'{INPUTPATH}/X.npy')
    y = np.load( f'{INPUTPATH}/y.npy')
    date = np.load( f'{INPUTPATH}/date.npy')
    
    
    gkf =  PurgedGroupTimeSeriesSplit(n_splits = FOLDS,  group_gap = GROUP_GAP)
    if MDL_NAME == 'autoencoder':
        model = autoencoder(input_size = X.shape[-1], output_size = y.shape[-1], noise=0.1).to(DEVICE)
    else:
        raise NameError('Model name is not aligned with the actual model.')
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'min',verbose=True,patience=5)
    logger.info(model)
    
    
    
    sts = time.time()
    learn_hist_list = []
    save_path_list = []
    for fold, (tr, vl) in enumerate(gkf.split(y, y, date)):
        logger.info('Fold : {}'.format(fold+1))

        X_tr, X_val = X[tr], X[vl]
        y_tr, y_val = y[tr], y[vl]
        trn_dat = CustomDataset(X_tr, y_tr)
        val_dat = CustomDataset(X_val, y_val)
        trn_loader = DataLoader(trn_dat , batch_size=BATCH_SIZE, shuffle=False)
        val_loader = DataLoader(val_dat , batch_size=BATCH_SIZE, shuffle=False)
        loaders = {'train':trn_loader, 'valid': val_loader}
        trained_model, learn_hist, save_path =\
            train_model(model, criterion, optimizer, scheduler, loaders, DEVICE, NUM_EPOCH, PATIANCE, \
                    MDL_PATH, MDL_NAME, VER, fold+1,logger)
        
        fig_path = f'{MDL_PATH}/{MDL_NAME}_{VER}/figures'
        if not os.path.exists(fig_path):
            os.mkdir(fig_path)
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        plt.plot(learn_hist.epoch, learn_hist.valid_loss, color = 'blue')
        ax2 = ax1.twinx()
        plt.plot(learn_hist.epoch, learn_hist.train_loss, color = 'red')
        ax1.set_ylabel('Valid Loss')
        ax2.set_ylabel('Train Loss')
        plt.xlabel('Epochs')
        plt.title('Learning Curve')
        
        fig.savefig(fig_path+f'/learning_hist_fold{fold+1}.png')
        learn_hist['Fold'] = fold+1
        learn_hist_list.append(learn_hist)
        save_path_list.append(save_path)
    hist_path = f'{MDL_PATH}/{MDL_NAME}_{VER}/history'
    if not os.path.exists(hist_path):
                        os.mkdir(hist_path)
    all_hist = pd.concat(learn_hist_list, axis=0)
    all_hist.reset_index(inplace=True, drop=True)
    all_hist.to_csv(f'{MDL_PATH}/{MDL_NAME}_{VER}//history/{MDL_NAME}_learning_history.csv', index=False)
    ed = time.time()
    logger.info('Training process takes {:.2f} min.'.format((ed-sts)/60))
    
    
    
    @njit(fastmath = True)
    def utility_score_numba(date, weight, resp, action):
        Pi = np.bincount(date, weight * resp * action)
        t = np.sum(Pi) / np.sqrt(np.sum(Pi ** 2)) * np.sqrt(250 / len(Pi))
        u = min(max(t, 0), 6) * np.sum(Pi)
        return u
    
    
    model_list  = glob.glob(f'{MDL_PATH}/{MDL_NAME}_{VER}/*.pth')
    loop = int(np.round(len(X)/BATCH_SIZE))
    pred_all = np.array([])
    for n in tqdm(range(loop)):
        x_tt = X[BATCH_SIZE*n:BATCH_SIZE*(n+1),:]
        if np.isnan(x_tt[:, 1:].sum()):
            x_tt[:, 1:] = np.nan_to_num(x_tt[:, 1:]) + np.isnan(x_tt[:, 1:]) * f_mean
        pred = 0.0
        X_test = torch.FloatTensor(x_tt).to(DEVICE)
        for mdl in model_list:
            load_weights = torch.load(mdl)
            model.load_state_dict(load_weights)
            model.eval()
            pred += model(X_test).cpu().detach().numpy() / FOLDS
        if len(pred_all) == 0:
            pred_all = pred.copy()
        else:
            pred_all = np.vstack([pred_all, pred]).copy()

    
    weight = np.load( f'{INPUTPATH}/weight.npy' )
    resp = np.load( f'{INPUTPATH}/resp.npy')
    action = np.where(pred_all[:,0] >= THRESHOLD, 1, 0).astype(int).copy()
    logger.info(utility_score_numba(date, weight, resp, action))
    

            
if __name__ == "__main__":
    main()
#     args = get_args()
#     with open(args.config_path, 'r') as f:
#         config = yaml.safe_load(f)
#     print(config['USE_FINETUNE'])
#     EXT = config['EXT']
#     logging.basicConfig(level = 'DEBUG', filename=f'../logs/{EXT}.log')
#     logger = logging.getLogger('Log')
#     logger.info(config)
#     logger.info(sys.argv)

    
    #with open(f'../logs/param_{EXT}.json', mode='w') as f:
    #    f.write(str(config))