import argparse
import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import h5py
import scipy.io
from torch.autograd import Variable

import laina_models

from dense_estimation.losses import RMSLoss, RelLoss, TestingLosses, Log10Loss
from dense_estimation.datasets.image_utils import BilinearResize

def test(model_data_path, images, targets):

    # Default input size
    height = 240
    width = 320
    channels = 3
    batch_size = 1


    # Create a placeholder for the input image
    input_node = tf.placeholder(tf.float32, shape=(batch_size, height, width, channels))

    # Construct the network
    net = laina_models.ResNet50UpProj({'data': input_node}, batch_size)

    testing_multi_criterion = TestingLosses([RMSLoss(), RelLoss(), Log10Loss()])

    losses = np.zeros(3)
    resize = BilinearResize(0.5)

    with tf.Session() as sess:

        # Load the converted parameters
        print('Loading the model')
        net.load(model_data_path, sess)

        uninitialized_vars = []
        for var in tf.global_variables():
            try:
                sess.run(var)
            except tf.errors.FailedPreconditionError:
                uninitialized_vars.append(var)

        init_new_vars_op = tf.variables_initializer(uninitialized_vars)
        sess.run(init_new_vars_op)

        for i in range(images.shape[0]):
            image = np.expand_dims(resize(images[i, :, :, :]), 0)
            target = np.expand_dims(targets[i, :, :, :], 0)

            pred = sess.run(net.get_output(), feed_dict={input_node: image})
            target_pt = Variable(torch.Tensor(target).permute(0, 3, 1, 2)) # NHWC -> NCHW
            pred_pt = Variable(torch.Tensor(pred).permute(0, 3, 1, 2)) # NHWC -> NCHW

            upsample = nn.UpsamplingBilinear2d(size=target_pt.size()[2:])
            pred_pt = upsample(pred_pt)
            losses += testing_multi_criterion(pred_pt, target_pt).data.numpy()

            if i < 5:
                fig = plt.figure()
                plt.imshow(pred_pt.data.numpy()[0,0,:,:], cmap='gray')
            print("{}/{}".format(i, images.shape[0]))
        plt.show()

        losses /= len(images)
        print(losses)


def load_testing_data(root):
    folder = os.path.join(root, 'nyu_depth_v2')
    nyud_file_path = os.path.join(folder, 'nyu_depth_v2_labeled.mat')
    splits_file_path = os.path.join(folder, 'splits.mat')
    nyud_dict = h5py.File(nyud_file_path, 'r')
    splits_dict = scipy.io.loadmat(splits_file_path)
    images = np.asarray(nyud_dict['images'], dtype=np.float32)
    depths = np.asarray(nyud_dict['depths'], dtype=np.float32)

    # NCWH -> NHWC
    images = np.transpose(images, (0, 3, 2, 1))
    depths = np.transpose(np.expand_dims(depths, 1), (0, 3, 2, 1))

    indices = splits_dict['testNdxs'][:, 0] - 1
    images = np.take(images, indices, axis=0)
    depths = np.take(depths, indices, axis=0)
    return images, depths


def main():
    images, depths = load_testing_data('./datasets')
    #images = np.random.rand(5, 480, 640, 3)
    #depths = np.random.rand(5, 480, 640, 1)
    pred = test('./downloads/NYU_ResNet-UpProj.npy', images, depths)


if __name__ == '__main__':
    main()
