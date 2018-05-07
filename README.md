# Monocular Depth Prediction

This repository contains a unofficial PyTorch implementation of a monocular depth prediction model described in 
"[Deeper Depth Prediction with Fully Convolutional Residual Networks]
(https://arxiv.org/abs/1606.00373)". 
For the official models, see the
[FCRN-DepthPrediction](https://github.com/iro-cp/FCRN-DepthPrediction) repository.
This implementation supports data pre-processing, training from scratch, and evaluation. The code currently only supports the NYU Depth v2 dataset, but it should be easy to add other datasets.

## Setup (Python 3)

### Install prerequisites
* install pytorch (see pytorch.org)
* install tensorflow (for tensorboard visualization only - no gpu support required)
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
* open `localhost:6006` in a browser

### Continue training from checkpoint
Checkpoints are stored after each epoch.

* `python train.py --ex my_test --epochs 80 --lr 0.01`
* `python train.py --ex my_test --epochs 50 --lr 0.003`

### View all training options
* `python train.py --help`
