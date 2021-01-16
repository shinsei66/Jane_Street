import glob
import os
import json
import gc
import sys
import time
import yaml
import argparse
import logging
from tqdm import tqdm
import subprocess
from utils import get_args
import torch
print(torch.__version__)


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
    COMMENT = config['COMMENT']
    
    format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level = logging.INFO,format=format_str, filename=f'../logs/upload_log_{MDL_NAME}_{VER}_{EXT}.log')
    logger = logging.getLogger('Log')
    
    ##https://ryoz001.com/1154.html
    # コンソール画面用ハンドラー設定
    # ハンドラーのログレベルを設定する (INFO以上を出力する)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.INFO)
    # logger と コンソール用ハンドラーの関連付け
    logger.addHandler(consoleHandler)
    logger.info(config)
    logger.info(sys.argv)
    
    VER = (VER + '_' + EXT)
    
    model_path = f'{MDL_PATH}/{MDL_NAME}_{VER}'
    logger.info(model_path)
    
    model_list = glob.glob(model_path+'/*.pth')
    
    data_json =  {
        "title": "Jane-Street",
        "id": "shinsei66/Jane-Street",
        "subtitle": "",
        "description": "",
        "isPrivate": True,
        "licenses": [
            {
                "name": "unknown" 
            }
        ],
        "keywords": [],
        "collaborators": [],
        "data": [

        ]
    }
    
    data_list = []
    for mdl in model_list:
        mdl_nm = mdl.replace(model_path+'/', '')
        mdl_size = os.path.getsize(mdl) 
        data_dict = {
            "description": COMMENT,
            "name": f'{mdl_nm}',
            "totalBytes": mdl_size,
            "columns": []
        }
        data_list.append(data_dict)
    data_json['data'] = data_list
    
    logger.info(data_json)
    
    with open(model_path+'/dataset-metadata.json', 'w') as f:
        json.dump(data_json, f)
    
    
    
    script = f'kaggle datasets version -p {model_path} -m \"{MDL_NAME}_{VER}\"'
    script = ['kaggle',  'datasets', 'version', '-p', f'{model_path}' , '-m' , f'\"{MDL_NAME}_{VER}\"']
    #script = 'ls'
    logger.info(script)
    logger.info(subprocess.check_output(script))
    
    
if __name__ == "__main__":
    main()
    