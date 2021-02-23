import glob
#import cupy as cp
import warnings
warnings.filterwarnings('ignore')
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
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
# print(torch.__version__)
import matplotlib.pyplot as plt
# from numba import njit
from janest_model import CustomDataset, train_lgb, ResNetModel, SmoothBCEwLogits, utility_score_bincount, TransformerModel, log_evaluation, autoencoder2
from utils import PurgedGroupTimeSeriesSplit, get_args
import lightgbm as lgb



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
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LR =config['LR']
    WEIGHT = config['WEIGHT']
    MDL_PATH  =config['MDL_PATH']
    MDL_NAME =config['MDL_NAME']
    VER = config['VER']
    THRESHOLD = config['THRESHOLD']
    DATAVER = config['DATAVER']
    
    format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level = logging.INFO,format=format_str, filename=f'../logs/{MDL_NAME}_{VER}_{EXT}.log')
    logger = logging.getLogger('Log')
    logger.info(config)
    logger.info(sys.argv)
    
    f_mean = np.load( f'{INPUTPATH}/f_mean_{DATAVER}.npy')
    X = np.load( f'{INPUTPATH}/X_{DATAVER}.npy')
    y = np.load( f'{INPUTPATH}/y_{DATAVER}.npy')
    date = np.load( f'{INPUTPATH}/date_{DATAVER}.npy')
    weight = np.load( f'{INPUTPATH}/weight_{DATAVER}.npy' )
    resp = np.load( f'{INPUTPATH}/resp_{DATAVER}.npy')
    
    if TRAINING:
        gkf =  PurgedGroupTimeSeriesSplit(n_splits = FOLDS,  group_gap = GROUP_GAP)
        for fold, (tr, vl) in enumerate(gkf.split(y, y, date)):

            pass

        logger.info('Train Data Date: {}'.format(np.unique(date[tr])))
        logger.info('Valid Data Date: {}'.format(np.unique(date[vl])))
        
        
    VER = (VER + '_' + EXT)
    model = autoencoder2(input_size = X.shape[-1], output_size = y.shape[-1],
                        noise=0.2
                       ).to(DEVICE)
    logger.info(f'fine tuning initial weight path is : {WEIGHT}')
    model.load_state_dict(torch.load(WEIGHT))
    logger.info(model)
    model.eval()
    
    X_tr, X_val = X[tr], X[vl]
    y_tr, y_val = y[tr], y[vl]
    
    trn_dat = CustomDataset(X_tr, y_tr)
    val_dat = CustomDataset(X_val, y_val)
    trn_loader = DataLoader(trn_dat , batch_size=len(X_tr), shuffle=False)
    val_loader = DataLoader(val_dat , batch_size=len(X_val), shuffle=False)
    for i, data in enumerate(trn_loader):
        train = model.feature(data['x'].to(DEVICE)).cpu().detach().numpy()
    for i, data in enumerate(val_loader):
        valid = model.feature(data['x'].to(DEVICE)).cpu().detach().numpy()
    param_lgb = {
           'num_leaves': 512,
          'min_child_samples': 79,
          'objective': 'binary',
          'learning_rate': LR,
          "boosting_type": "gbdt",
          "subsample_freq": 3,
          "subsample": 0.9,
          "bagging_seed": 66,
          "metric": 'binary_logloss',
          "verbosity": -1,
          'reg_alpha': 0.3,
          'reg_lambda': 0.3,
          'colsample_bytree': 0.9,
         }
    
    save_path = f'{MDL_PATH}/{MDL_NAME}_{VER}/'
    for n in range(5):
        trn_data = lgb.Dataset(train, label=y_tr[:,n])
        val_data = lgb.Dataset(valid, label=y_val[:,n])
        xgb_data = {'trn':trn_data, 'val':val_data}

        if MDL_NAME == 'lightgbm':
            lgbmodel = train_lgb(param_lgb, xgb_data, NUM_EPOCH, 100, PATIANCE, True,logger, save_path, n)

        else:
            raise NameError('Model name is not aligned with the actual model.')
        y_pred = np.zeros(len(valid))
        y_pred += lgbmodel.predict(valid)

    
    action = np.where(y_pred> THRESHOLD, 1, 0).astype(int).copy()

    if np.sum(action)>0:
        date_vl = date[vl].copy()
        weight_vl = weight[vl].copy()
        resp_vl = resp[vl].copy()
        action_ans_vl = np.where(y[vl,0]> 0, 1, 0).astype(int).copy()
        cv_score = utility_score_bincount(date_vl[:action.shape[0]] , weight_vl[:action.shape[0]] , resp_vl[:action.shape[0]] , action)
        max_score = utility_score_bincount(date_vl , weight_vl , resp_vl , action_ans_vl )
#                 print('CV score is {}, Max score is {}, return ration is {:.1f} '.format(cv_score, max_score, 100*(cv_score/max_score)))
        logger.info('CV score is {}, Max score is {}, return ratio is {:.1f} '.format(cv_score, max_score, 100*(cv_score/max_score)))
    else:
        raise ZeroDivisionError
    

            
if __name__ == "__main__":
    main()
