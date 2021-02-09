import pandas as pd
import numpy as np
import yaml
import glob
import logging
import os
import sys
import gc
import time
from utils import get_args



def main():
    sts = time.time()
    args = get_args()
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    INPUTPATH = config['INPUTPATH']
    DATAVER = config['DATAVER']
    format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level = logging.INFO,format=format_str, filename=f'../logs/preprocess_{DATAVER}.log')
    logger = logging.getLogger(__name__)
    logger.info(config)
    logger.info(sys.argv)
    
    
    train = pd.read_csv(f'{INPUTPATH}/train.csv')
    test_df = pd.read_csv(f'{INPUTPATH}/example_test.csv')
    pred_df  = pd.read_csv(f'{INPUTPATH}/example_sample_submission.csv')
    df_feat = pd.read_csv(f'{INPUTPATH}/features.csv')



    train = train.query('date > 85').reset_index(drop = True) 

    train.fillna(train.mean(),inplace=True)
    train = train.query('weight > 0').reset_index(drop = True)
    
    for i in range(28):
        train[f'tag_{i}_features_mean']=train[df_feat[df_feat.loc[ :, 'tag_'+str(i)]==True].feature.to_list()].mean(axis=1)
    features = [c for c in train.columns if 'feature' in c]
    x_tt = train.loc[:, features].values
    train['features_41_42_43'] = x_tt[:, 41] + x_tt[:, 42] + x_tt[:, 43]
    train['features_1_2'] = x_tt[:, 1] / (x_tt[:, 2] + 1e-5)
    logger.info(train.shape)
    features = [c for c in train.columns if 'feature' in c]
    logger.info('{} features are generated.'.format(len(features)))

    resp_cols = ['resp_1', 'resp_2', 'resp_3', 'resp', 'resp_4']

    X = train[features].values
    y = np.stack([(train[c] > 0).astype('int') for c in resp_cols]).T
#     y = np.stack([train[c]  for c in resp_cols]).T
    #train['action'] =  np.stack([(train[c] > 0).astype('int') for c in resp_cols]).T
    f_mean = np.mean(train[features[1:]].values,axis=0)
    date = train['date'].values
    weight = train['weight'].values
    resp = train['resp'].values


    np.save( f'{INPUTPATH}/f_mean_{DATAVER}.npy',f_mean)
    np.save( f'{INPUTPATH}/X_{DATAVER}.npy',X)
    np.save( f'{INPUTPATH}/y_{DATAVER}.npy',y)
    np.save( f'{INPUTPATH}/date_{DATAVER}.npy',date)
    np.save( f'{INPUTPATH}/weight_{DATAVER}.npy',weight )
    np.save( f'{INPUTPATH}/resp_{DATAVER}.npy',resp)
    
    ed = time.time()
    logger.info('Data prepared in {:.2f} sec'.format(ed-sts))

if __name__ == "__main__":
    
    main()