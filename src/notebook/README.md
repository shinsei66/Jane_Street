# File Contents & Examples

- janest_model.py: model class
```
from janest_model import CustomDataset, train_model, autoencoder2, ResNetModel, SmoothBCEwLogits, utility_score_bincount, TransformerModel
```
- utils.py: utilities
```
from utils import PurgedGroupTimeSeriesSplit, get_args
```
- preprocess_02.py: create data set
```
python preprocess_02.py  parameters_pre_05.yaml
```
- upload_model.py: create or update kaggle dataset
```
python upload_model.py parameters_cv_ae.yaml
```
- train_ho_*.py: hold out training
```
python train_ho_ae.py parameters_cv_ae.yaml
```
- train_cv_*.py:cross validation training
```
python train_cv_ae.py parameters_cv_ae.yaml
```
- *.yaml:parameter setting