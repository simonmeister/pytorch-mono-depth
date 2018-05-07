import os
import shutil
import numpy as np
import tensorflow as tf
import torch
import matplotlib.pyplot as plt


class TensorBoardLogger():
    def __init__(self, log_dir, visualizer, max_testing_images=3,
                 testing_loss_names=[], run_options=None,
                 starting_epoch=0):
        self.summary_writer = tf.summary.FileWriter(log_dir)
        self.sess = tf.Session()
        self.visualizer = visualizer
        self.image_placeholders = []
        self.testing_loss_placeholders = []

        for name in visualizer.names:
            p = tf.placeholder(tf.float32, name=name)
            self.image_placeholders.append(p)
            tf.summary.image(name, p, collections=['testing_images'],
                             max_outputs=max_testing_images)

        self.training_loss_placeholder = tf.placeholder(tf.float32, name='training_loss')
        tf.summary.scalar('training/loss', self.training_loss_placeholder,
                          collections=['training_loss'])

        self.learning_rate_placeholder = tf.placeholder(tf.float32, name='training_lr')
        tf.summary.scalar('training/learning_rate', self.learning_rate_placeholder,
                          collections=['training_lr'])

        for name in testing_loss_names:
            p = tf.placeholder(tf.float32, name=name)
            self.testing_loss_placeholders.append(p)
            tf.summary.scalar('testing/'+name, p, collections=['testing_losses'])

        if os.path.isdir(log_dir):
            shutil.rmtree(log_dir)
        os.mkdir(log_dir)

        if False: #run_options is not None: # TODO not out yet in pre-built tensorflow
            run_options_placeholder = tf.placeholder(tf.string,
                                                     name='training_run_options')
            tf.summary.text('training/run_options',
                            self.training_loss_placeholder,
                              collections=['training_run_options'])

            feed_dict = {run_options_placeholder: run_options}
            self._eval_and_add_summary(feed_dict, 'training_run_options',
                                       starting_epoch)

    def _eval_and_add_summary(self, feed_dict, key, step):
        summaries = tf.get_collection(key)
        summary_ = tf.summary.merge(summaries)
        summary = self.sess.run(summary_, feed_dict=feed_dict)
        self.summary_writer.add_summary(summary, step)

    def log_training_loss(self, iteration, loss, learning_rate):
        feed_dict = {self.training_loss_placeholder: loss}
        self._eval_and_add_summary(feed_dict, 'training_loss', iteration)
        feed_dict = {self.learning_rate_placeholder: learning_rate}
        self._eval_and_add_summary(feed_dict, 'training_lr', iteration)

    def log_testing_images(self, epoch, input, outputs, target):
        arrays = self.visualizer(input, outputs, target)
        images = visuals_to_numpy(arrays)

        feed_dict = dict(zip(self.image_placeholders, images))
        self._eval_and_add_summary(feed_dict, 'testing_images', epoch)

    def log_testing_losses(self, epoch, losses):
        feed_dict = dict(zip(self.testing_loss_placeholders, losses))
        self._eval_and_add_summary(feed_dict, 'testing_losses', epoch)

    def close():
        self.summary_writer.close()
        self.sess.close()


def visuals_to_numpy(arrays):
    images = []
    for x in arrays:
        if isinstance(x, tuple):
            x, cmap_fn = x
        else:
            cmap_fn = None
        x_np = np.transpose(x.numpy(), (0, 2, 3, 1))
        if cmap_fn is not None:
            x_np = cmap_fn(x_np[:, :, :, 0])
        images.append(x_np)
    return images


class BasicVisualizer():
    """Visualizes a single image element input, output and target."""
    names = ['image', 'prediction', 'truth']

    def __call__(self, input, outputs, target):
        return [input, outputs[0], target]


class DistributionVisualizer():
    """Visualizes output distribution mean and variance."""
    #names = ['image', 'mean', 'variance', 'truth']
    names = ['mean', 'variance', 'error']

    def __init__(self, distribution):
        self.distribution = distribution

    def __call__(self, input, outputs, target):
        d = self.distribution(*outputs)

        # TODO create distribution plot

        #return [target,
        #        d.mean,
        #        (2*d.variance, plt.cm.jet),
        #        (torch.abs(target-d.mean), plt.cm.jet)]
        return [d.mean, d.variance, torch.abs(target-d.mean)]
