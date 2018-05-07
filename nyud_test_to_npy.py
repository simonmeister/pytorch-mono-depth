import os
import shutil
from oct2py import octave
from PIL import Image
import numpy as np
import scipy.ndimage
import os
import scipy.io
import h5py

from dense_estimation.datasets.util import maybe_download


NYUD_URL = 'http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat'
NYUD_SPLITS_URL = 'http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat'


def save_npy(source_dir, target_dir):
    if not os.path.isdir(source_dir):
        os.makedirs(source_dir)
    nyud_file_path = os.path.join(source_dir, 'nyu_depth_v2_labeled.mat')
    splits_file_path = os.path.join(source_dir, 'splits.mat')

    maybe_download(NYUD_URL, nyud_file_path)
    maybe_download(NYUD_SPLITS_URL, splits_file_path)

    print("Loading dataset: NYU Depth V2")
    nyud_dict = h5py.File(nyud_file_path, 'r')
    splits_dict = scipy.io.loadmat(splits_file_path)

    images = np.asarray(nyud_dict['images'], dtype=np.float32)
    depths = np.asarray(nyud_dict['depths'], dtype=np.float32)

    # convert to NCHW arrays
    images = images.swapaxes(2, 3)
    depths = np.expand_dims(depths.swapaxes(1, 2), 1)

    #if split == 'train':
    #    indices = splits_dict['trainNdxs'][:, 0] - 1
    #else:
    #    indices = splits_dict['testNdxs'][:, 0] - 1
    indices = splits_dict['testNdxs'][:, 0] - 1

    images = np.take(images, indices, axis=0)
    depths = np.take(depths, indices, axis=0)

    npy_folder = os.path.join(target_dir, 'npy')
    if os.path.isdir(npy_folder):
        shutil.rmtree(npy_folder)
    os.makedirs(npy_folder)

    np.save(os.path.join(npy_folder, 'images.npy'), images)
    np.save(os.path.join(npy_folder, 'depths.npy'), depths)


if __name__ == '__main__':
    save_npy('/home/smeister/work/depth-prediction/datasets/nyu_depth_v2',
             '/home/smeister/datasets/nyu_depth_v2/labeled')
