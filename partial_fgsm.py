# Modifying the iterative attack.
# - See the _top_k_abs_grads method for an example of what to do.
# - If you introduce some new controllable parameter, define it in
#   config.json, then add it to the tunable_params list and it will be made
#   available to you in params to use. Also add it to input_parser.py under
#   the FLAGS.DEFINE_x section as the appropriate type so it can be
#   parameterized via command line.
# - Find the method you want to edit in the PartialFgsmAttack class.
# - Edit that method as needed. You just need to access x (batched images),
#   grad (the grad of loss wrt x), and params (any parameters defined in
#   config.json).
# - The method needs to return the perturbed image (x by default).
# - Don't worry about bounds checking on the L-infinity distance. The clip
#   function handles it all, so just perturb as much as you want and the
#   result will always be a valid image.

# Running:
# - Don't directly run this file. Run main.py.

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
from run_attack import run_attack

from sklearn import preprocessing

tf.logging.set_verbosity(tf.logging.ERROR)

class PartialFgsmAttack(LinfPGDAttack):
  def perturb(self, x_nat, y, sess, params):
    """Given a set of examples (x_nat, y), returns a set of adversarial
       examples within epsilon of x_nat in l_infinity norm."""
    # x_nat is shape (batch_size, 784), 784 because the images are 28x28.
    if self.rand:
      x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
    else:
      x = np.copy(x_nat)

    for i in range(self.k):
      grad = sess.run(self.grad, feed_dict={self.model.x_input: x,
                                            self.model.y_input: y})

      # Runs the attack itself and perturbs the images in x.
      x = self._run_attack(x, grad, params)

      x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon) 
      x = np.clip(x, 0, 1) # ensure valid pixel range

    return x

  def _run_attack(self, x, grad, params):
    # Dispatches to the desired attack type to run.
    if params['partial_method'] == 'top_k_abs_grads':
      return self._top_k_abs_grads(x, grad, params)
    elif params['partial_method'] == 'thresh_abs_grads':
      return self._thresh_abs_grads(x, grad, params)
    elif params['partial_method'] == 'distrib_grads':
      return self._distrib_grads(x, grad, params)
    elif params['partial_method'] == 'clipped_pixels':
      return self._clipped_pixels(x, grad, params)
    elif params['partial_method'] == 'top_k_distrib_grads':
      return self._top_k_distrib_grads(x, grad, params)
    else:
      err = 'Method not implemented: {]'.format(params['partial_method'])
      raise RuntimeError(err)

  def _top_k_abs_grads(self, x, grad, params):
    # This method will change the top k gradient values for x.
    # x = image in batch form, shape (batch_size, flattened image size)
    # grad = gradient of loss wrt x, shape (batch_size, flattened image size)
    # params = dictionary of parameters, where the keys are defined in
    #          config.json under tunable params and the values are the values
    #          actually defined in config.json under the corresponding key.
    top_k = params['top_grads']
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

    x += params['a'] * np.sign(thresholded_grads)
    return x

  def _thresh_abs_grads(self, x, grad, params):
    # This method will change x if its gradient exceeds grad_thresh defined in config.json.
    # x = image in batch form, shape (batch_size, flattened image size)
    # grad = gradient of loss wrt x, shape (batch_size, flattened image size)
    # params = dictionary of parameters, where the keys are defined in
    #          config.json under tunable params and the values are the values
    #          actually defined in config.json under the corresponding key.
    # Take the absolute values of the gradient values
    grad_abs = np.abs(grad)
    # Create a mask for all absolute gradient values >= grad_thresh set to 1 and everything else set to 0
    mask = grad_abs >= params['grad_thresh']
    thresholded_grads = np.zeros_like(grad)
    thresholded_grads[mask] = grad[mask]
    # Update x for pixels whose gradient is >= grad_thresh
    x += params['a'] * np.sign(thresholded_grads)
    return x

  def _distrib_grads(self, x, grad, params):
    # This method will change x based on the distribution of its gradient.
    # x = image in batch form, shape (batch_size, flattened image size)
    # grad = gradient of loss wrt x, shape (batch_size, flattened image size)
    # params = dictionary of parameters, where the keys are defined in
    #          config.json under tunable params and the values are the values
    #          actually defined in config.json under the corresponding key.
    # Normalize the gradient values
    norm_grads = preprocessing.normalize(grad, norm='l2')
    # Take steps of size 'a' distributed according to the normalized gradient
    x += params['a'] * norm_grads
    return x

  def _top_k_distrib_grads(self, x, grad, params):
    # This method will change the top k normalized gradient values of x.
    # x = image in batch form, shape (batch_size, flattened image size)
    # grad = gradient of loss wrt x, shape (batch_size, flattened image size)
    # params = dictionary of parameters, where the keys are defined in
    #          config.json under tunable params and the values are the values
    #          actually defined in config.json under the corresponding key.
    top_k = params['top_grads']
    # TODO: there has to be a better way to do this.
    # Normalize the gradient values
    norm_grads = preprocessing.normalize(grad, norm='l2')
    # Take steps of size 'a' distributed according to the normalized gradient
    norm_grad_abs = np.abs(norm_grads)
    # Get sorting for max absolute value of the gradient.
    ranking = np.argsort(norm_grad_abs)
    # Get kth largest (-top_k) index so we can create a boolean mask array
    # and create a threshold mask for pixels that we should update.
    top_idx = ranking[:, -top_k]
    abs_thresholds = norm_grad_abs[np.arange(norm_grad_abs.shape[0]), top_idx]
    abs_thresholds_full = np.repeat(abs_thresholds, repeats=norm_grad_abs.shape[1])
    abs_thresholds_full = abs_thresholds_full.reshape(norm_grad_abs.shape)
    update_mask = norm_grad_abs >= abs_thresholds_full
    # Only choose the gradients that are in the top_k.
    norm_thresholded_grads = np.zeros_like(grad)
    norm_thresholded_grads[update_mask] = norm_grads[update_mask]

    # Update top pixel values by step size 'a' times the normalized gradient value
    x += params['a'] * norm_thresholded_grads
    return x

  def _clipped_pixels(self, x, grad, params):
    # TODO: implement me.
    # This method will change x the number of times k defined in config.json.
    # x = image in batch form, shape (batch_size, flattened image size)
    # grad = gradient of loss wrt x, shape (batch_size, flattened image size)
    # params = dictionary of parameters, where the keys are defined in
    #          config.json under tunable params and the values are the values
    #          actually defined in config.json under the corresponding key.
    raise RuntimeError('Not implemented yet')


if __name__ == '__main__':
  import sys
  import numpy as np
  from model import Model
  model = Model()
  attack = PartialFgsmAttack(model,
                             0,
                             0,
                             0,
                             True,
                             'xent')
  x = np.array([[0.1, 0.2], [0.3, -1.0]])
  grads = np.array([[0, 0.5], [0.5, 0.6]])
  params = {'top_grads':1,
            'a': 5}
  attack._top_k_distrib_grads(x, grads, params)
  sys.exit(0)


  import json
  import sys
  import math
  import tensorflow as tf
  import datetime
  import pathlib

  from tensorflow.examples.tutorials.mnist import input_data

  from model import Model

  # Timestamp for unique logging.
  timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

  # Command line args, use to override config.json inputs.
  flags = tf.app.flags
  flags.DEFINE_string('model_dir', None, 'Where the trained model is stored.')
  flags.DEFINE_integer('top_grads', None, 'Number of top grads to threshold')
  flags.DEFINE_integer('k', None, 'How many training steps to take')
  flags.DEFINE_float('a', None, 'Fixed training step size')
  flags.DEFINE_string('loss_func', None, 'The loss function to lose (?)')
  flags.DEFINE_boolean('delete_attacks', False, 'Delete attack files')
  FLAGS = tf.app.flags.FLAGS

  with open('config.json') as config_file:
    config = json.load(config_file)

  # Config import.
  top_grads = config['top_grads']
  epsilon = config['epsilon']
  k = config['k']
  a = config['a']
  random_start = config['random_start']
  loss_func = config['loss_func']
  model_dir = config['model_dir']
  delete_attacks = config['delete_attacks']
  save_dir = config['save_dir']

  # Custom flag parsing, override anything in config for parameterization.
  if FLAGS.model_dir:
    model_dir = FLAGS.top_grads
  if FLAGS.top_grads:
    top_grads = FLAGS.top_grads
  if FLAGS.k:
    k = FLAGS.k
  if FLAGS.a:
    a = FLAGS.a
  if FLAGS.loss_func:
    loss_func = FLAGS.loss_func
  if FLAGS.delete_attacks:
    delete_attacks = FLAGS.delete_attacks

  # Create output path directory.
  pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

  # Create the unique name for the save file.
  model_name = config['model_dir'].split('/')[1]
  params = {'model': model_name,
            'top_grads': top_grads,
            'epsilon': epsilon,
            'k': k,
            'a': a,
            'random_start': random_start,
            'loss_func': loss_func}
  sorted_items = sorted(params.items(), key=lambda x: x[0])
  items = [str(v) for k, v in sorted_items]
  k_v_items = ['{}:{}'.format(k, v) for k, v in sorted_items]
  full_items = '-'.join(items)
  adv_path = '{}/{}-{}-attack.npy'.format(save_dir, timestamp, full_items)
  y_pred_path = '{}/{}-{}-ypred.npy'.format(save_dir, timestamp, full_items)
  summary_path = '{}/{}-{}-summary.txt'.format(save_dir, timestamp, full_items)

  model_file = tf.train.latest_checkpoint(model_dir)
  if model_file is None:
    print('No model found')
    sys.exit()

  model = Model()
  attack = PartialFgsmAttack(model,
                             epsilon,
                             k,
                             a,
                             random_start,
                             loss_func)
  saver = tf.train.Saver()

  mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

  #with tf.Session() as sess:
  sess = tf.Session()
  # Restore the checkpoint
  saver.restore(sess, model_file)

  # Iterate over the samples batch-by-batch
  num_eval_examples = config['num_eval_examples']
  eval_batch_size = config['eval_batch_size']
  num_batches = int(math.ceil(num_eval_examples / eval_batch_size))

  x_adv = [] # adv accumulator

  # print('Iterating over {} batches'.format(num_batches))
  print('Generating adversarial examples: {}'.format(adv_path))

  for ibatch in range(num_batches):
    bstart = ibatch * eval_batch_size
    bend = min(bstart + eval_batch_size, num_eval_examples)
    # print('batch size: {}'.format(bend - bstart))

    x_batch = mnist.test.images[bstart:bend, :]
    y_batch = mnist.test.labels[bstart:bend]

    x_batch_adv = attack.perturb(x_batch, y_batch, sess, top_grads)

    x_adv.append(x_batch_adv)

  # print('Storing examples')
  x_adv = np.concatenate(x_adv, axis=0)
  if not delete_attacks:
    print('saving')
    np.save(adv_path, x_adv)
  # print('Examples stored in {}'.format(adv_path))
  sess.close()
  tf.reset_default_graph()

  print('Evaluating results: {}'.format(y_pred_path))
  eval_results_path = None if delete_attacks else y_pred_path
  accuracy = run_attack(model_file, x_adv, epsilon, eval_results_path)
  print('Accuracy: {}'.format(accuracy))

  with open(summary_path, 'w') as writefile:
    for line in k_v_items:
      writefile.write(line + '\n')
    writefile.write('accuracy: {}'.format(accuracy))
