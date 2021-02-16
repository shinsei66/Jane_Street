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
#from tqdm.notebook import tqdm
from tqdm import tqdm
from torch.utils.data import DataLoader
print(torch.__version__)
import matplotlib.pyplot as plt
from numba import njit
from janest_model import CustomDataset, train_model, autoencoder2, ResNetModel, SmoothBCEwLogits, utility_score_bincount, TransformerModel
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
    DEVICE = torch.cuda.set_device(0)
#     DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(DEVICE)
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
    if MDL_NAME == 'transformer':
        model = TransformerModel(input_size = X.shape[-1], output_size = y.shape[-1], batch_size = BATCH_SIZE).to(DEVICE)
    else:
        raise NameError('Model name is not aligned with the actual model.')
#     criterion = nn.L1Loss()
#     criterion = nn.BCELoss()
    label_smoothing = 1e-2
    criterion = SmoothBCEwLogits(smoothing=label_smoothing)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR,
        weight_decay=1e-5
    )
    scheduler = ReduceLROnPlateau(optimizer, 'min',0.5,verbose=True,patience=50)
#     scheduler = CosineAnnealingLR(optimizer, T_max=250, eta_min=1e-8, last_epoch=-1, verbose=True)
    logger.info(model)
    
    VER = (VER + '_' + EXT)

    
    if TRAINING:
        sts = time.time()
        learn_hist_list = []
        save_path_list = []
        for fold, (tr, vl) in enumerate(gkf.split(y, y, date)):
            logger.info('Fold : {}'.format(fold+1))

            X_tr, X_val = X[tr], X[vl]
            y_tr, y_val = y[tr], y[vl]
            date_vl = date[vl].copy()
            weight_vl = weight[vl].copy()
            resp_vl = resp[vl].copy()
            action_ans_vl = np.where(y[vl,0]> THRESHOLD, 1, 0).astype(int).copy()
#             print('{}  {}  {}  {}'.format(date_vl.shape, weight_vl.shape, resp_vl.shape, action_ans_vl.shape))
            data = {'date': date_vl, 'weight': weight_vl, 'resp': resp_vl, 'ans':action_ans_vl,
                   'x_vl': X_val}
            
            trn_dat = CustomDataset(X_tr, y_tr)
            val_dat = CustomDataset(X_val, y_val)
            trn_loader = DataLoader(trn_dat , batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
            val_loader = DataLoader(val_dat , batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
            loaders = {'train':trn_loader, 'valid': val_loader}
            trained_model, learn_hist, save_path =\
                train_model(model, criterion, optimizer, scheduler, loaders, DEVICE, NUM_EPOCH, PATIANCE, \
                        MDL_PATH, MDL_NAME, VER, fold+1,logger, data)

            fig_path = f'{MDL_PATH}/{MDL_NAME}_{VER}/figures'
            logger.info(save_path)
            trained_model.load_state_dict(torch.load(save_path))
            trained_model.eval()
            preds = []       
            for i, data in tqdm(enumerate(val_loader)):
                x = data['x'].to(DEVICE)
                with torch.no_grad():
                    outputs = trained_model(x)
                    preds.append(outputs.sigmoid().detach().cpu().numpy())

            pred_all  = np.concatenate(preds)

            #action = np.where(pred_all[:,0] >= THRESHOLD, 1, 0).astype(int).copy()
            date_vl = date[vl].copy()
            weight_vl = weight[vl].copy()
            resp_vl = resp[vl].copy()
            action_ans_vl = np.where(y[vl,0]> THRESHOLD, 1, 0).astype(int).copy()
            action = np.where(np.mean(pred_all, axis=1)> THRESHOLD, 1, 0).astype(int).copy()
            cv_score = utility_score_bincount(date_vl[:action.shape[0]]  , weight_vl[:action.shape[0]], resp_vl[:action.shape[0]]  , action)
            max_score = utility_score_bincount(date_vl , weight_vl , resp_vl , action_ans_vl )
            logger.info('Fold {}: CV score is {}, Max score is {}, return ratio is {:.1f} '\
                        .format(fold+1, cv_score, max_score, 100*(cv_score/max_score)))
            
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
        all_hist.to_csv(f'{MDL_PATH}/{MDL_NAME}_{VER}/history/{MDL_NAME}_learning_history.csv', index=False)
        ed = time.time()
        logger.info('Training process takes {:.2f} min.'.format((ed-sts)/60))

            
    if not TRAINING:   
        gkf =  PurgedGroupTimeSeriesSplit(n_splits = FOLDS,  group_gap = GROUP_GAP)
        for fold, (tr, vl) in enumerate(gkf.split(y, y, date)):
            X_tr, X_val = X[tr], X[vl]
            y_tr, y_val = y[tr], y[vl]
            trn_dat = CustomDataset(X_tr, y_tr)
            val_dat = CustomDataset(X_val, y_val)
            trn_loader = DataLoader(trn_dat , batch_size=BATCH_SIZE, shuffle=False)
            val_loader = DataLoader(val_dat , batch_size=BATCH_SIZE, shuffle=False)
            model_list  = glob.glob(f'{MDL_PATH}/{MDL_NAME}_{VER}/*.pth')

            preds = []   
            
            for i, data in tqdm(enumerate(val_loader)):
                x = data['x'].to(DEVICE)
                outputs = np.zeros((len(x), 5))
                with torch.no_grad():
                    for mdl in model_list:
                        load_weights = torch.load(mdl)
                        model.load_state_dict(load_weights)
                        model.eval()
#                         outputs += model(x).sigmoid().detach().cpu().numpy()/len(model_list)
                        outputs += model(x).detach().cpu().numpy()/len(model_list)
                    preds.append(outputs)

            pred_all  = np.concatenate(preds)
            

#             action = np.where(pred_all[:,0] >= THRESHOLD, 1, 0).astype(int).copy()
            action = np.where(np.mean(pred_all, axis=1)> THRESHOLD, 1, 0).astype(int).copy()
            if np.sum(action)>0:
                date_vl = date[vl].copy()
                weight_vl = weight[vl].copy()
                resp_vl = resp[vl].copy()
                action_ans_vl = np.where(y[vl,0]> THRESHOLD, 1, 0).astype(int).copy()
                cv_score = utility_score_bincount(date_vl[:action.shape[0]] , weight_vl[:action.shape[0]] , resp_vl[:action.shape[0]] , action)
                max_score = utility_score_bincount(date_vl , weight_vl , resp_vl , action_ans_vl )
#                 print('CV score is {}, Max score is {}, return ration is {:.1f} '.format(cv_score, max_score, 100*(cv_score/max_score)))
                logger.info('CV score is {}, Max score is {}, return ratio is {:.1f} '.format(cv_score, max_score, 100*(cv_score/max_score)))
            else:
                raise ZeroDivisionError


if __name__ == "__main__":
    main()
