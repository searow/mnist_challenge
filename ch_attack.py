"""Runs CleverHans attacks on the Madry Lab MNIST challenge model

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
from ch_model_interface import MadryMNIST
from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import BasicIterativeMethod
from cleverhans.attacks import MadryEtAl
from cleverhans.attacks import MomentumIterativeMethod
from cleverhans.utils_mnist import data_mnist

import sys as _sys
import input_parser
import pathlib
import math
import numpy as np
from cleverhans.utils import _ArgsWrapper

FLAGS = flags.FLAGS

def get_examples(sess, x, y, adv_x, X_test=None, Y_test=None, 
                  feed=None, args=None):
  args = _ArgsWrapper(args or {})
  adv_x_list = np.array([])

  with sess.as_default():
    nb_batches = int(math.ceil(float(len(X_test)) / args.batch_size))
    assert nb_batches * args.batch_size >= len(X_test)

    X_cur = np.zeros((args.batch_size,) + X_test.shape[1:],
                     dtype=X_test.dtype)
    for batch in range(nb_batches):
      if batch % 100 == 0 and batch > 0:
        _logger.debug("Batch " + str(batch))

      # Must not use the `batch_indices` function here, because it
      # repeats some examples.
      # It's acceptable to repeat during training, but not eval.
      start = batch * args.batch_size
      end = min(len(X_test), start + args.batch_size)

      # The last batch may be smaller than all others. This should not
      # affect the accuarcy disproportionately.
      cur_batch_size = end - start
      X_cur[:cur_batch_size] = X_test[start:end]
      feed_dict = {x: X_cur}
      if feed is not None:
        feed_dict.update(feed)
      batch_adv = adv_x.eval(feed_dict=feed_dict)

      if(len(adv_x_list) == 0):
        adv_x_list = np.copy(batch_adv)
      else:
        adv_x_list = np.concatenate(
          (adv_x_list, batch_adv[:cur_batch_size]), axis=0)

    assert end >= len(X_test)

  return adv_x_list

def get_attack_examples(argv, parser):
  checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

  if checkpoint is None:
    raise ValueError("Couldn't find latest checkpoint in " +
                     FLAGS.checkpoint_dir)

  train_start = 0
  train_end = 60000
  test_start = 0
  test_end = 10000
  X_train, Y_train, X_test, Y_test = data_mnist(train_start=train_start,
                                                train_end=train_end,
                                                test_start=test_start,
                                                test_end=test_end)

  assert Y_train.shape[1] == 10

  # NOTE: for compatibility with Madry Lab downloadable checkpoints,
  # we cannot enclose this in a scope or do anything else that would
  # change the automatic naming of the variables.
  model = MadryMNIST()

  x_input = tf.placeholder(tf.float32, shape=[None, 784])
  x_image = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
  y = tf.placeholder(tf.float32, shape=[None, 10])

  if FLAGS.attack_type == 'fgsm':
    fgsm = FastGradientMethod(model)
    fgsm_params = {'eps': parser.epsilon, 'clip_min': 0., 'clip_max': 1.}
    adv_x = fgsm.generate(x_image, **fgsm_params)
  elif FLAGS.attack_type == 'bim':
    bim = BasicIterativeMethod(model)
    bim_params = {'eps': tf.cast(parser.epsilon, tf.float32), 
                  'clip_min': 0., 
                  'clip_max': 1.,
                  'nb_iter': parser.params['k'],
                  'eps_iter': parser.params['a']}
    adv_x = bim.generate(x_image, **bim_params)
  elif FLAGS.attack_type == 'pgd':
    pgd = MadryEtAl(model)
    pgd_params = {'eps': tf.cast(parser.epsilon, tf.float32), 
                  'clip_min': 0., 
                  'clip_max': 1.,
                  'nb_iter': parser.params['k'],
                  'eps_iter': parser.params['a'],
                  'ord': np.inf,
                  'rand_init': parser.random_start}
    adv_x = pgd.generate(x_image, **pgd_params)
  elif FLAGS.attack_type == 'moment':
    mmt = MomentumIterativeMethod(model)
    mmt_params = {'eps': tf.cast(parser.epsilon, tf.float32), 
                  'clip_min': 0., 
                  'clip_max': 1.,
                  'nb_iter': parser.params['k'],
                  'eps_iter': parser.params['a'],
                  'ord': np.inf,
                  'decay_factor': 1.0}
    adv_x = mmt.generate(x_image, **mmt_params)
  else:
    raise ValueError(FLAGS.attack_type)

  preds_adv = model.get_probs(adv_x)

  saver = tf.train.Saver()

  with tf.Session() as sess:
    # Restore the checkpoint
    saver.restore(sess, checkpoint)
    eval_par = {'batch_size': FLAGS.batch_size}

    adv_x_list = \
      get_examples(sess, x_image, y, adv_x, X_test, Y_test, args=eval_par)

  adv_x_list = np.squeeze(np.reshape(adv_x_list, (10000,28*28,1)))
  return adv_x_list

def ch_attack(parser):
  dirs = ['models', 'secret']
  default_checkpoint_dir = os.path.join(*dirs)

  if('batch_size' not in tf.flags.FLAGS.__flags):
    flags.DEFINE_integer('batch_size', 128, "batch size")
    flags.DEFINE_float(
        'label_smooth', 0.1, ("Amount to subtract from correct label "
                              "and distribute among other labels"))
    flags.DEFINE_string(
        'attack_type', parser.params['partial_method'],
        ("Attack type: 'fgsm'->fast gradient sign"
                                "method, 'bim'->'basic iterative method'"))
    flags.DEFINE_string('checkpoint_dir', default_checkpoint_dir,
                        'Checkpoint directory to load')

  argv = flags.FLAGS(_sys.argv, known_only=True)
  return get_attack_examples(argv, parser)

if __name__ == '__main__':
  # Parses the config.json, FLAGS, and overrides.
  parser = input_parser.Parser()

  # Create output path directory.
  pathlib.Path(parser.save_dir).mkdir(parents=True, exist_ok=True)

  parser.params['partial_method'] = 'fgsm'
  print(ch_attack(parser).shape)