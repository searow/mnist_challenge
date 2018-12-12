import datetime
import json
import math
import tensorflow as tf

class Parser(object):
  def __init__(self):
    flags = self._parse_flags()
    config = self._get_config_file()
    self._set_tunable_params(flags, config)
    self._set_fixed_params(flags, config)
    self._set_save_file_names()

  def get_summary_data(self):
    summary = {k: v for k, v in self.params.items()} # copy
    summary['model'] = self.model_name
    return summary

  def _parse_flags(self):
    # Command line args, use to override config.json inputs.
    flags = tf.app.flags
    flags.DEFINE_string('model_dir', None, 'Where the trained model is stored.')
    flags.DEFINE_integer('k', None, 'How many training steps to take')
    flags.DEFINE_float('a', None, 'Fixed training step size')
    flags.DEFINE_float('decay_factor', None, 'Decay factor for momentum term')
    flags.DEFINE_string('loss_func', None, 'The loss function to lose (?)')
    flags.DEFINE_boolean('delete_attacks', False, 'Delete attack files')
    # Method-specific flags
    flags.DEFINE_string('partial_method', None, 'The method in partial_fgsm'
                                                ' that should be run')
    flags.DEFINE_float('grad_thresh', None, '')
    flags.DEFINE_integer('top_grads', None, 'Number of top grads to threshold')
    flags.DEFINE_integer('random_inits', None, 'How man inits to do')
    FLAGS = tf.app.flags.FLAGS
    return FLAGS

  def _get_config_file(self):
    # Parses the base config file to get defaults for each input and 
    # still be compatible with pgd_attack.py.
    with open('config.json') as config_file:
      config = json.load(config_file)

    return config

  def _set_tunable_params(self, flags, config):
    # Sets self.params for every tunable parameter that we defined.
    tunable = config['tunable_params']
    self.params = {}
    # For some reason tensorflow updates the flags after this error is raised...
    try:
      flags.FLAGS
    except:
      pass
    for key in tunable:
      config_default = config[key]
      if flags[key].value:
        self.params[key] = flags[key].value
      else:
        self.params[key] = config[key]

  def _set_fixed_params(self, flags, config):
    # Sets any fixed params that we need for every model.
    self.timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    self.epsilon = config['epsilon']
    self.random_start = config['random_start']
    self.loss_func = config['loss_func']
    self.save_dir = config['save_dir']
    self.model_dir = config['model_dir']
    self.model_name = config['model_dir'].split('/')[1]
    self.delete_attacks = config['delete_attacks']
    if flags.model_dir:
      self.model_dir = flags.model_dir
    if flags.delete_attacks:
      self.delete_attacks = flags.delete_attacks
    if flags.loss_func:
      self.loss_func = flags.loss_func
    self.num_eval_examples = config['num_eval_examples']
    self.eval_batch_size = config['eval_batch_size']
    self.num_batches = int(math.ceil(
        self.num_eval_examples / self.eval_batch_size))

  def _set_save_file_names(self):
    model_name = self.model_name
    params = self.params
    params['model'] = self.model_name
    params['epsilon'] = self.epsilon
    sorted_items = sorted(params.items(), key=lambda x: x[0])
    items = [str(v) for k, v in sorted_items]
    k_v_items = ['{}:{}'.format(k, v) for k, v in sorted_items]
    full_items = '-'.join(items)
    self.adv_path = '{}/{}-{}-attack.npy'.format(self.save_dir,
                                                 self.timestamp,
                                                 full_items)
    self.y_pred_path = '{}/{}-{}-ypred.npy'.format(self.save_dir,
                                                   self.timestamp,
                                                   full_items)
    self.summary_path = '{}/{}-{}-summary.txt'.format(self.save_dir,
                                                      self.timestamp,
                                                      full_items)
    self.heatmap_path = '{}/{}-{}-heatmap.txt'.format(self.save_dir,
                                                      self.timestamp,
                                                      full_items)
