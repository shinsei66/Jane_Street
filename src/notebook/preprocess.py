import pandas as pd
import numpy as np
import glob
#import cupy as cp
import os
import gc
import time
#import torch
#import torchvision
#from torch.optim.lr_scheduler import ReduceLROnPlateau
#from torch import nn
#import torch.nn.functional as F
#from tqdm.notebook import tqdm
#from torch.utils.data import DataLoader
#print(torch.__version__)
#import matplotlib.pyplot as plt
#from numba import njit
#%matplotlib inline
#from janest_model import MLPNet , CustomDataset, train_model
#from utils import PurgedGroupTimeSeriesSplit



INPUTPATH = '../../input'



def main():
    train = pd.read_parquet(f'{INPUTPATH}/train.parquet')
    test_df = pd.read_csv(f'{INPUTPATH}/example_test.csv')
    pred_df  = pd.read_csv(f'{INPUTPATH}/example_sample_submission.csv')



    train = train.query('date > 85').reset_index(drop = True) 
    print(train.shape)
    train.fillna(train.mean(),inplace=True)
    train = train.query('weight > 0').reset_index(drop = True)
    train['action'] =  \
    (  (train['resp_1'] > 0.00001 ) & \
       (train['resp_2'] > 0.00001 ) & \
       (train['resp_3'] > 0.00001 ) & \
       (train['resp_4'] > 0.00001 ) & \
       (train['resp'] > 0.00001 )   ).astype('int')

    features = [c for c in train.columns if 'feature' in c]

    resp_cols = ['resp_1', 'resp_2', 'resp_3', 'resp', 'resp_4']

    X = train[features].values
    y = np.stack([(train[c] > 0.000001).astype('int') for c in resp_cols]).T
    f_mean = np.mean(train[features[1:]].values,axis=0)


    np.save( f'{INPUTPATH}/f_mean.npy',f_mean)
    np.save( f'{INPUTPATH}/X.npy',X)
    np.save( f'{INPUTPATH}/y.npy',y)

if __name__ == "__main__":
    sts = time.time()
    main()
    ed = time.time()
    print('Data prepared in {:.2f} sec'.format(ed-sts))