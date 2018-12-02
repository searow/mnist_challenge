# Running new attacks:
# 1. Make sure the base attack is runnable by running pgd_attack.py. This
#    only generates the attack result npy file that has the images that
#    should be evaluated with run_attack.py.
# 2. Run this file which runs a default configuration in config.py.
# 3. To run parametric version, run this file but specify flag parameters
#    on the command line using --<flagname>=<flag>. See the flags we
#    parse in the main function. They'll override anything that is in
#    config.py but continue to use the ones not specified.
# 4. Summary results (params + final accuracy) are saved in the attack
#    directory specified in config file.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import input_parser
import json
import numpy as np
import pathlib
import sys
import tensorflow as tf
import ch_attack

from tensorflow.examples.tutorials.mnist import input_data

from model import Model
from partial_fgsm import PartialFgsmAttack
from run_attack import run_attack

def partial_attack(parser):
  # Create the model and the custom attack.
  model = Model()

  attack = PartialFgsmAttack(model,
                             parser.epsilon,
                             parser.params['k'],
                             parser.params['a'],
                             parser.random_start,
                             parser.loss_func)

  # Load MNIST data.
  mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

  # Restore the checkpoint (pretrained model).
  sess = tf.Session()
  saver = tf.train.Saver()
  saver.restore(sess, model_file)

  # Generate batches for images.
  num_eval_examples = parser.num_eval_examples
  eval_batch_size = parser.eval_batch_size
  num_batches = parser.num_batches

  x_adv = [] # adv accumulator

  print('Attacking: {}'.format(parser.adv_path))

  for ibatch in range(num_batches):
    bstart = ibatch * eval_batch_size
    bend = min(bstart + eval_batch_size, num_eval_examples)

    x_batch = mnist.test.images[bstart:bend, :]
    y_batch = mnist.test.labels[bstart:bend]

    # parser.params has all the info needed for the attack. It reflects the
    # tunable params in config.json.
    x_batch_adv = attack.perturb(x_batch, y_batch, sess, parser.params)

    x_adv.append(x_batch_adv)
  x_adv = np.concatenate(x_adv, axis=0)

  # Save the attacks if necessary.
  if not parser.delete_attacks:
    print('Saving: {}'.format(parser.adv_path))
    np.save(parser.adv_path, x_adv)

  return x_adv

if __name__ == '__main__':
  # Parses the config.json, FLAGS, and overrides.
  parser = input_parser.Parser()

  # Create output path directory.
  pathlib.Path(parser.save_dir).mkdir(parents=True, exist_ok=True)

  # Import the trained madry model.
  model_file = tf.train.latest_checkpoint(parser.model_dir)
  if model_file is None:
    print('No model found')
    sys.exit()

  if(parser.params['partial_method'] in {'fgsm', 'bim'}):
    x_adv = ch_attack.ch_attack(parser)
  else:
    x_adv = partial_attack(parser)
  
  # Reset tensorflow. Important otherwise the eval won't run properly!
  tf.Session().close()
  tf.reset_default_graph()

  # Run the evaluation on the newly generated adversarial images.
  print('Evaluating results: {}'.format(parser.y_pred_path))
  eval_results_path = None if parser.delete_attacks else parser.y_pred_path
  accuracy = run_attack(model_file,
                        x_adv,
                        parser.epsilon,
                        eval_results_path)
  print('Accuracy: {}'.format(accuracy))

  # Print the reports to the output path.
  with open(parser.summary_path, 'w') as writefile:
    summary = parser.get_summary_data()
    summary['accuracy'] = accuracy
    json.dump(summary, writefile, indent=4)