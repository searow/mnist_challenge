"""
Implementation of attack methods. Running this file as a program will
apply the attack to the model specified by the config file and store
the examples in an .npy file.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from pgd_attack import LinfPGDAttack


class PartialFgsmAttack(LinfPGDAttack):
  def perturb(self, x_nat, y, sess, top_k):
    """Given a set of examples (x_nat, y), returns a set of adversarial
       examples within epsilon of x_nat in l_infinity norm."""
    if self.rand:
      x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
    else:
      x = np.copy(x_nat)

    for i in range(self.k):
      grad = sess.run(self.grad, feed_dict={self.model.x_input: x,
                                            self.model.y_input: y})

      # TODO: there has to be a better way to do this.
      # Get sorting for max absolute value of the gradient.
      grad_abs = np.abs(grad)
      ranking = np.argsort(grad_abs)
      # Get kth largest (-top_k) index so we can create a boolean mask array
      # and create a threshold mask for pixels that we should update.
      top_idx = ranking[:, -top_k]
      abs_thresholds = grad_abs[np.arange(grad_abs.shape[0]), top_idx]
      abs_thresholds_full = np.repeat(abs_thresholds, repeats=grad_abs.shape[1])
      abs_thresholds_full = abs_thresholds_full.reshape(grad_abs.shape)
      update_mask = grad_abs >= abs_thresholds_full
      # Only choose the gradients that are in the top_k.
      thresholded_grads = np.zeros_like(grad)
      thresholded_grads[update_mask] = grad[update_mask]

      x += self.a * np.sign(thresholded_grads)

      x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon) 
      x = np.clip(x, 0, 1) # ensure valid pixel range

    return x


if __name__ == '__main__':
  import json
  import sys
  import math

  from tensorflow.examples.tutorials.mnist import input_data

  from model import Model

  with open('config.json') as config_file:
    config = json.load(config_file)

  top_grads = config['top_grads']
  model_file = tf.train.latest_checkpoint(config['model_dir'])
  if model_file is None:
    print('No model found')
    sys.exit()

  model = Model()
  attack = PartialFgsmAttack(model,
                             config['epsilon'],
                             config['k'],
                             config['a'],
                             config['random_start'],
                             config['loss_func'])
  saver = tf.train.Saver()

  mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

  with tf.Session() as sess:
    # Restore the checkpoint
    saver.restore(sess, model_file)

    # Iterate over the samples batch-by-batch
    num_eval_examples = config['num_eval_examples']
    eval_batch_size = config['eval_batch_size']
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))

    x_adv = [] # adv accumulator

    print('Iterating over {} batches'.format(num_batches))

    for ibatch in range(num_batches):
      bstart = ibatch * eval_batch_size
      bend = min(bstart + eval_batch_size, num_eval_examples)
      print('batch size: {}'.format(bend - bstart))

      x_batch = mnist.test.images[bstart:bend, :]
      y_batch = mnist.test.labels[bstart:bend]

      x_batch_adv = attack.perturb(x_batch, y_batch, sess, top_grads)

      x_adv.append(x_batch_adv)

    print('Storing examples')
    path = config['store_adv_path']
    x_adv = np.concatenate(x_adv, axis=0)
    np.save(path, x_adv)
    print('Examples stored in {}'.format(path))
