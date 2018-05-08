# Monocular Depth Prediction

This repository contains a unofficial PyTorch implementation of a monocular depth prediction model described in 
["Deeper Depth Prediction with Fully Convolutional Residual Networks"](https://arxiv.org/abs/1606.00373) by [Iro Laina](http://campar.in.tum.de/Main/IroLaina) and others. 
For the official models, see the
[FCRN-DepthPrediction](https://github.com/iro-cp/FCRN-DepthPrediction) repository.
This implementation supports data pre-processing, training from scratch, and evaluation. The code currently only supports the NYU Depth v2 dataset, but it should be easy to add other datasets.

Note that there is some code to support uncertainty (variance) prediction, however there are some dependencies missing from this repo and i didn't have time to document this. You don't need to worry about this code and can always leave the `--dist` argument set to `''` to use the code for standard depth prediction.

### TODO
- upload evaluation performance numbers on NYU Depth
- document test.py script

### License
This project is licensed under the MIT License (refer to the LICENSE file for details).

## Setup (Python 3)

### Install prerequisites
* install [pytorch](https://pytorch.org/)
* install [tensorflow](https://www.tensorflow.org/) (for tensorboard visualization only - no gpu support required). The easiest way is to run `pip install tensorflow`.
* install other python packages: `pip install scipy matplotlib h5py`
* install matlab (the pre-processing script depends on the NYU Depth v2 matlab toolbox)

### Prepare datasets
* `python nyud_test_to_npy.py` (modify the paths in that file to point to correct dirs)
* download the NYU Depth v2 raw dataset (~400GB) and the toolbox from https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html.
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
