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
    
    
    train = pd.read_parquet(f'{INPUTPATH}/train.parquet')
    test_df = pd.read_csv(f'{INPUTPATH}/example_test.csv')
    pred_df  = pd.read_csv(f'{INPUTPATH}/example_sample_submission.csv')
    df_feat = pd.read_csv(f'{INPUTPATH}/features.csv')



    train = train.query('date > 85').reset_index(drop = True) 
    print(train.shape)
    train.fillna(train.mean(),inplace=True)
    train = train.query('weight > 0').reset_index(drop = True)
    
    for i in range(28):
        train[f'tag_{i}_features_mean']=train[df_feat[df_feat.loc[ :, 'tag_'+str(i)]==True].feature.to_list()].mean(axis=1)
    
    features = [c for c in train.columns if 'feature' in c]
    logger.info('{} features are generated.'.format(len(features)))

    resp_cols = ['resp_1', 'resp_2', 'resp_3', 'resp', 'resp_4']

    X = train[features].values
    y = np.stack([(train[c] > 0).astype('int') for c in resp_cols]).T
#     y = np.stack([train[c]  for c in resp_cols]).T
    train['action'] =  np.stack([(train[c] > 0).astype('int') for c in resp_cols]).T
    f_mean = np.mean(train[features[1:]].values,axis=0)
    date = train['date'].values
    weight = train['weight'].values
    resp = train['resp'].values


    np.save( f'{INPUTPATH}/f_mean.npy',f_mean)
    np.save( f'{INPUTPATH}/X.npy',X)
    np.save( f'{INPUTPATH}/y.npy',y)
    np.save( f'{INPUTPATH}/date.npy',date)
    np.save( f'{INPUTPATH}/weight.npy',weight )
    np.save( f'{INPUTPATH}/resp.npy',resp)
    
    ed = time.time()
    logger.info('Data prepared in {:.2f} sec'.format(ed-sts))

if __name__ == "__main__":
    
    main()