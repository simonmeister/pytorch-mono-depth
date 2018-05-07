# Depth Prediction #

## Setup (Python 3)
* install pytorch (see pytorch.org)
* install tensorflow (no gpu support needed)
* install python packages: `scipy matplotlib h5py`

### Prepare datasets
* `python nyud_test_to_npy.py` (modify the paths in that file to point to correct dirs)
* download NYU Depth v2 raw dataset (~400GB) toolbox
* generate training dataset with matlab - see process_raw.m
* `python nyud_raw_train_to_npy.py` (modify the paths in that file to point to correct dirs)
* modify raw_root in train.py and test.py to point to correct dir


## Usage examples

### Train and view results
* `python train.py --ex my_test`
* `tensorboard logdir=log/my_test`
* open 'localhost:6006' in browser

### Continue training from checkpoint
Checkpoints are stored after each epoch.

* `python train.py --ex my_test --epochs 80 --lr 0.01`
* `python train.py --ex my_test --epochs 50 --lr 0.003`

### View all training options
* `python train.py --help`
