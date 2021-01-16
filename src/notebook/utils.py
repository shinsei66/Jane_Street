from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_categorical_dtype

def reduce_mem_usage(df, use_float16=False):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    for col in df.columns:
        if is_datetime(df[col]) or is_categorical_dtype(df[col]):
            # skip datetime type or categorical type
            continue
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        #else:
            #df[col] = df[col].astype('category')
    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df


import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection._split import _BaseKFold, indexable, _num_samples
from sklearn.utils.validation import _deprecate_positional_args


# modified code for group gaps; source
# https://github.com/getgaurav2/scikit-learn/blob/d4a3af5cc9da3a76f0266932644b884c99724c57/sklearn/model_selection/_split.py#L2243
class PurgedGroupTimeSeriesSplit(_BaseKFold):
    """Time Series cross-validator variant with non-overlapping groups.
    Allows for a gap in groups to avoid potentially leaking info from
    train into test if the model has windowed or lag features.
    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals according to a
    third-party provided group.
    In each split, test indices must be higher than before, and thus shuffling
    in cross validator is inappropriate.
    This cross-validation object is a variation of :class:`KFold`.
    In the kth split, it returns first k folds as train set and the
    (k+1)th fold as test set.
    The same group will not appear in two different folds (the number of
    distinct groups has to be at least equal to the number of folds).
    Note that unlike standard cross-validation methods, successive
    training sets are supersets of those that come before them.
    Read more in the :ref:`User Guide <cross_validation>`.
    Parameters
    ----------
    n_splits : int, default=5
        Number of splits. Must be at least 2.
    max_train_group_size : int, default=Inf
        Maximum group size for a single training set.
    group_gap : int, default=None
        Gap between train and test
    max_test_group_size : int, default=Inf
        We discard this number of groups from the end of each train split
    """

    @_deprecate_positional_args
    def __init__(self,
                 n_splits=5,
                 *,
                 max_train_group_size=np.inf,
                 max_test_group_size=np.inf,
                 group_gap=None,
                 verbose=False
                 ):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.max_train_group_size = max_train_group_size
        self.group_gap = group_gap
        self.max_test_group_size = max_test_group_size
        self.verbose = verbose

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        if groups is None:
            raise ValueError(
                "The 'groups' parameter should not be None")
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        group_gap = self.group_gap
        max_test_group_size = self.max_test_group_size
        max_train_group_size = self.max_train_group_size
        n_folds = n_splits + 1
        group_dict = {}
        u, ind = np.unique(groups, return_index=True)
        unique_groups = u[np.argsort(ind)]
        n_samples = _num_samples(X)
        n_groups = _num_samples(unique_groups)
        for idx in np.arange(n_samples):
            if (groups[idx] in group_dict):
                group_dict[groups[idx]].append(idx)
            else:
                group_dict[groups[idx]] = [idx]
        if n_folds > n_groups:
            raise ValueError(
                ("Cannot have number of folds={0} greater than"
                 " the number of groups={1}").format(n_folds,
                                                     n_groups))

        group_test_size = min(n_groups // n_folds, max_test_group_size)
        group_test_starts = range(n_groups - n_splits * group_test_size,
                                  n_groups, group_test_size)
        for group_test_start in group_test_starts:
            train_array = []
            test_array = []

            group_st = max(0, group_test_start - group_gap - max_train_group_size)
            for train_group_idx in unique_groups[group_st:(group_test_start - group_gap)]:
                train_array_tmp = group_dict[train_group_idx]
                
                train_array = np.sort(np.unique(
                                      np.concatenate((train_array,
                                                      train_array_tmp)),
                                      axis=None), axis=None)

            train_end = train_array.size
 
            for test_group_idx in unique_groups[group_test_start:
                                                group_test_start +
                                                group_test_size]:
                test_array_tmp = group_dict[test_group_idx]
                test_array = np.sort(np.unique(
                                              np.concatenate((test_array,
                                                              test_array_tmp)),
                                     axis=None), axis=None)

            test_array  = test_array[group_gap:]
            
            
            if self.verbose > 0:
                    pass
                    
            yield [int(i) for i in train_array], [int(i) for i in test_array]

            
import argparse
#https://rightcode.co.jp/blog/information-technology/pytorch-yaml-optimizer-parameter-management-simple-method-complete
def get_args():
    # 引数の導入
    parser = argparse.ArgumentParser(description='Cross Validation Train')
    parser.add_argument('config_path', type=str, help='Setting parameter(.yaml)')
    args = parser.parse_args()
    return args


import subprocess
import glob
import json
import os
def upload_to_kaggle(
                     
                     title: str, 
                     k_id: str,  
                     path: str, 
                     comments: str,
                     update:bool,
                     logger=None,
                     extension = '.pth',
                     subtitle='', 
                     description="",
                     isPrivate = True,
                     licenses = "unknown" ,
                     keywords = [],
                     collaborators = []
                     ):
    '''
    >> upload_to_kaggle(title, k_id, path,  comments, update)
    
    Arguments
    =========
     title: the title of your dataset.
     k_id: kaggle account id.
     path: non-default string argument of the file path of the data to be uploaded.
     comments:non-default string argument of the comment or the version about your upload.
     logger: logger object if you use logging, default is None.
     extension: the file extension of model weight files, default is ".pth"
     subtitle: the subtitle of your dataset, default is empty string.
     description: dataset description, default is empty string.
     isPrivate: boolean to show wheather to make the data public, default is True.
     licenses = the licenses description, default is "unkown"; must be one of /
     ['CC0-1.0', 'CC-BY-SA-4.0', 'GPL-2.0', 'ODbL-1.0', 'CC-BY-NC-SA-4.0', 'unknown', 'DbCL-1.0', 'CC-BY-SA-3.0', 'copyright-authors', 'other', 'reddit-api', 'world-bank'] .
     keywords : the list of keywords about the dataset, default is empty list.
     collaborators: the list of dataset collaborators, default is empty list.
   '''
    model_list = glob.glob(path+f'/*{extension}')
    if len(model_list) == 0:
        raise FileExistsError('File does not exist, check the file extention is correct \
        or the file directory exist.')
    
    if path[-1] == '/':
        raise ValueError('Please remove the backslash in the end of the path')
    
    data_json =  {
        "title": title,
        "id": f"{k_id}/{title}",
        "subtitle": subtitle,
        "description": description,
        "isPrivate": isPrivate,
        "licenses": [
            {
                "name": licenses
            }
        ],
        "keywords": [],
        "collaborators": [],
        "data": [

        ]
    }
    
    data_list = []
    for mdl in model_list:
        mdl_nm = mdl.replace(path+'/', '')
        mdl_size = os.path.getsize(mdl) 
        data_dict = {
            "description": comments,
            "name": mdl_nm,
            "totalBytes": mdl_size,
            "columns": []
        }
        data_list.append(data_dict)
    data_json['data'] = data_list

    
    with open(path+'/dataset-metadata.json', 'w') as f:
        json.dump(data_json, f)
    
    script0 = ['kaggle',  'datasets', 'create', '-p', f'{path}' , '-m' , f'\"{comments}\"']
    script1 = ['kaggle',  'datasets', 'version', '-p', f'{path}' , '-m' , f'\"{comments}\"']

    #script0 = ['echo', '1']
    #script1 = ['echo', '2']

    if logger:    
        logger.info(data_json)
        
        if update:
            logger.info(script1)
            logger.info(subprocess.check_output(script1))
        else:
            logger.info(script0)
            logger.info(script1)
            logger.info(subprocess.check_output(script0))
            logger.info(subprocess.check_output(script1))
            
    else:
        print(data_json)
        
        if update:
            print(script1)
            print(subprocess.check_output(script1))
        else:
            print(script0)
            print(script1)
            print(subprocess.check_output(script0))
            print(subprocess.check_output(script1))