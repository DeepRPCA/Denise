"""Various models and functions to load them."""
# pylint: disable=invalid-name
import math
import os
import tempfile

from absl import flags
import numpy as np
import tensorflow as tf
from tensorflow import keras as k


flags.DEFINE_string(
    'weights_dir_path', None,
    'Path to directory where to load weights from.')
flags.DEFINE_string('model', None, 'Model topology to use.')
flags.DEFINE_string('activation', 'relu', 'Activation function to use.')
flags.DEFINE_bool('debug', False, 'To run with tf_debug')

FLAGS = flags.FLAGS


def _maybe_enable_debug():
  if not FLAGS.debug:
    return
  from tensorflow.python import debug as tf_debug  # pylint: disable=g-import-not-at-top
  tf.keras.backend.set_session(
      tf_debug.LocalCLIDebugWrapperSession(tf.Session()))


def minimalist1(tril_len, sigma_tril, output_size, trainable, N, forced_rank):
  del N
  del forced_rank
  nh = int(1E7 / (2 * (tril_len + output_size)))  # Number if hidden neurons.
  nn = k.layers.Dense(nh, activation=FLAGS.activation, name='L1',
                      trainable=trainable)(sigma_tril)
  logit = k.layers.Dense(output_size, name='logit')(nn)
  return logit


def minimalist2(tril_len, sigma_tril, output_size, trainable, N, forced_rank):
  del N
  del forced_rank
  nh = 2 * int(1E7 / (tril_len + output_size))  # Number if hidden neurons.
  nn = k.layers.Dense(nh//5, activation=FLAGS.activation, name='L1',
                      trainable=trainable)(sigma_tril)
  nn = k.layers.Dense(nh//8, activation=FLAGS.activation, name='L2',
                      trainable=trainable)(nn)
  nn = k.layers.Dense(nh, activation=FLAGS.activation, name='L3',
                      trainable=trainable)(nn)
  nn = k.layers.Dense(nh//8, activation=FLAGS.activation, name='L4',
                      trainable=trainable)(nn)
  nn = k.layers.Dense(nh//5, activation=FLAGS.activation, name='L5',
                      trainable=trainable)(nn)
  nn = k.layers.Dense(tril_len, activation=FLAGS.activation, name='L6',
                      trainable=trainable)(nn)
  logit = k.layers.Dense(output_size, name='logit')(nn)
  return logit


def get_topo_0(tril_len, sigma_tril, output_size, trainable, N, forced_rank):
  del N
  del forced_rank
  # What we started with, works well with 10*10.
  L1 = k.layers.Dense(tril_len*20, activation=FLAGS.activation, name='L1',
                      trainable=trainable)(sigma_tril)
  L2 = k.layers.Dense(tril_len*4, activation=FLAGS.activation, name='L2',
                      trainable=trainable)(L1)
  L3 = k.layers.Dense(tril_len*4, activation=FLAGS.activation, name='L3',
                      trainable=trainable)(L2)
  L4 = k.layers.Dense(tril_len*4, activation=FLAGS.activation, name='L4',
                      trainable=trainable)(L3)
  # No activation on purpose, we don't want M to be a single number.
  logit = k.layers.Dense(output_size, name='logit')(L4)
  return logit


def get_topo_1(tril_len, sigma_tril, output_size, trainable, N, forced_rank):
  del N
  del forced_rank
  L1 = k.layers.Dense(tril_len*10, activation=FLAGS.activation, name='L1',
                      trainable=trainable)(sigma_tril)
  L2 = k.layers.Dense(tril_len*4, activation=FLAGS.activation, name='L2',
                      trainable=trainable)(L1)
  L3 = k.layers.Dense(tril_len, activation=FLAGS.activation, name='L3',
                      trainable=trainable)(L2)
  L4 = k.layers.Dense(tril_len*8, activation=FLAGS.activation, name='L4',
                      trainable=trainable)(L3)
  # No activation on purpose, we don't want M to be a single number.
  logit = k.layers.Dense(output_size, name='logit')(L4)
  return logit


def get_topo_2(tril_len, sigma_tril, output_size, trainable, N, forced_rank):
  del N
  del forced_rank
  L1 = k.layers.Dense(tril_len, activation=FLAGS.activation, name='L1',
                      trainable=trainable)(sigma_tril)
  L2 = k.layers.Dense(tril_len*.5, activation=FLAGS.activation, name='L2',
                      trainable=trainable)(L1)
  L3 = k.layers.Dense(tril_len, activation=FLAGS.activation, name='L3',
                      trainable=trainable)(L2)
  # No activation on purpose, we don't want M to be a single number.
  logit = k.layers.Dense(output_size, name='logit')(L3)
  return logit


def get_topo_3(tril_len, sigma_tril, output_size, trainable, N, forced_rank):
  # What we started with, works well with 10*10.
  del N
  del forced_rank
  L1 = k.layers.Dense(tril_len*10, activation=FLAGS.activation, name='L1',
                      trainable=trainable)(sigma_tril)
  L2 = k.layers.Dense(tril_len*4, activation=FLAGS.activation, name='L2',
                      trainable=trainable)(L1)
  L3 = k.layers.Dense(tril_len*4, activation=FLAGS.activation, name='L3',
                      trainable=trainable)(L2)
  # No activation on purpose, we don't want M to be a single number.
  logit = k.layers.Dense(output_size, name='logit')(L3)
  return logit


def get_stacked_autoencoder(tril_len, sigma_tril, output_size, trainable,
                            N, forced_rank):
  # Decrease rank by 1 at each layer.
  del tril_len
  L = sigma_tril
  for i, rank in enumerate(range(N, forced_rank, -1)):
    L = k.layers.Dense(N * rank, activation=FLAGS.activation, name='L%d' % i,
                       trainable=trainable)(L)
  logit = k.layers.Dense(output_size, name='logit')(L)
  return logit


def autoencoder1(tril_len, sigma_tril, output_size, trainable, N, forced_rank):
  del N
  del forced_rank
  nn = k.layers.Dense(tril_len, activation=FLAGS.activation, name='L1',
                      trainable=trainable)(sigma_tril)
  nn = k.layers.Dense(tril_len, activation=FLAGS.activation, name='L2',
                      trainable=trainable)(nn)
  nn = k.layers.Dense(tril_len, activation=FLAGS.activation, name='L3',
                      trainable=trainable)(nn)
  nn = k.layers.Dense(tril_len, activation=FLAGS.activation, name='L4',
                      trainable=trainable)(nn)
  nn = k.layers.Dense(tril_len, activation=FLAGS.activation, name='L5',
                      trainable=trainable)(nn)
  nn = k.layers.Dense(output_size + tril_len, activation='elu', name='L6',
                      trainable=trainable)(nn)
  nn = k.layers.Dense(output_size + tril_len * .8, activation='elu', name='L7',
                      trainable=trainable)(nn)
  nn = k.layers.Dense(output_size + tril_len * .6, activation='elu', name='L8',
                      trainable=trainable)(nn)
  nn = k.layers.Dense(output_size + tril_len * .4, activation='elu', name='L9',
                      trainable=trainable)(nn)
  nn = k.layers.Dense(output_size + tril_len * .2, activation='elu', name='L10',
                      trainable=trainable)(nn)
  nn = k.layers.Dense(output_size, activation=FLAGS.activation, name='L11',
                      trainable=trainable)(nn)
  nn = k.layers.Dense(output_size*2, activation=FLAGS.activation, name='L12',
                      trainable=trainable)(nn)
  nn = k.layers.Dense(output_size, activation=FLAGS.activation, name='L13',
                      trainable=trainable)(nn)
  logit = k.layers.Dense(output_size, activation='linear', name='logit')(nn)
  return logit


def accordeon(tril_len, sigma_tril, output_size, trainable, N, forced_rank):
  del N
  del forced_rank
  nn = k.layers.Dense(tril_len, activation=FLAGS.activation, name='L1',
                      trainable=trainable)(sigma_tril)
  nn = k.layers.Dense(tril_len*2, activation=FLAGS.activation, name='L2',
                      trainable=trainable)(nn)
  nn = k.layers.Dense(tril_len*3, activation=FLAGS.activation, name='L3',
                      trainable=trainable)(nn)
  nn = k.layers.Dense(tril_len*2, activation=FLAGS.activation, name='L4',
                      trainable=trainable)(nn)
  nn = k.layers.Dense(tril_len, activation=FLAGS.activation, name='L5',
                      trainable=trainable)(nn)
  nn = k.layers.Dense(tril_len*2, activation=FLAGS.activation, name='L6',
                      trainable=trainable)(nn)
  nn = k.layers.Dense(tril_len*3, activation=FLAGS.activation, name='L7',
                      trainable=trainable)(nn)
  nn = k.layers.Dense(tril_len*2, activation=FLAGS.activation, name='L8',
                      trainable=trainable)(nn)
  nn = k.layers.Dense(output_size, activation=FLAGS.activation, name='L9',
                      trainable=trainable)(nn)
  logit = k.layers.Dense(output_size, name='logit')(nn)
  return logit


def get_non_initialized_model(N, forced_rank, shrink=False,
                              freeze_internal_layers=False):
  tril_len = int(N * (N + 1) / 2)  # input size of the neural network.
  output_size = N * forced_rank
  sigma = k.layers.Input(shape=(N, N), name='Sigma')
  sigma_flat = k.layers.Flatten(name='sigma_flat')(sigma)

  # Flatten lower triangle of sigma
  # Eg: if N = 3, gather_indices = [0, 3, 4, 6, 7, 8]
  gather_indices = np.arange(0, N**2).reshape([N, N])[np.tril_indices(N)]
  sigma_tril = k.layers.Lambda(
      lambda s: tf.gather(s, gather_indices, axis=-1),
      name='sigma_tril')(sigma_flat)

  model_getter = {
      'minimalist1': minimalist1,
      'minimalist2': minimalist2,
      'topo0': get_topo_0,
      'topo1': get_topo_1,
      'topo2': get_topo_2,
      'topo3': get_topo_3,
      'autoencoder1': autoencoder1,
      'stacked_autoencoder': get_stacked_autoencoder,
      'accordeon': accordeon,
  }[FLAGS.model]
  trainable = not freeze_internal_layers
  logit = model_getter(tril_len, sigma_tril, output_size, trainable,
                       N=N, forced_rank=forced_rank)

  # Neural network post processing:
  M = k.layers.Reshape((N, forced_rank), name='M')(logit)
  def mmt(m):
    return tf.matmul(m, m, transpose_b=True)
  L = k.layers.Lambda(mmt, name='L')(M)

  # # Experimental
  # # Rectify L so it sticks to sigma if where close to it, zeroing
  # # corresponding part of S.
  # epsilon = 0.20  # Should be set to same value used for eval.
  # def rectify_fn(l_sigma):
  #   l, sigma = l_sigma
  #   l_diff = sigma - l
  #   close_to_sigma = tf.logical_and(tf.less(l_diff, epsilon),
  #                                   tf.greater(l_diff, -epsilon))
  #   return tf.keras.backend.switch(close_to_sigma, sigma, l)
  # Lr = k.layers.Lambda(rectify_fn, name='rectify')([L, sigma])
  # L = Lr

  S = k.layers.Subtract(name='S')([sigma, L])

  if shrink:
    lambda_ = k.backend.constant(1 / math.sqrt(N), shape=(N, N))
    zeros = k.backend.constant(0., shape=(N, N))
    def shrink_f(S):
      return k.backend.sign(S) * k.backend.maximum(
          zeros, k.backend.abs(S) - lambda_)
    S = k.layers.Lambda(shrink_f)(S)

  model = k.models.Model(inputs=sigma, outputs=[S, L])
  return model


def _get_weights_path(N, forced_rank):
  weights_dir = FLAGS.weights_dir_path
  if not weights_dir:
    weights_dir = tempfile.mkdtemp()
  tf.io.gfile.makedirs(weights_dir)
  return os.path.join(weights_dir,
                      'N%s_k%s_weights.tf' % (N, forced_rank))


def get_model(N, forced_rank, shrink, raise_if_no_weights=False,
              freeze_internal_layers=False):
  """Returns path to weights file and model."""
  _maybe_enable_debug()
  model = get_non_initialized_model(N, forced_rank, shrink,
                                    freeze_internal_layers)
  weights_path = _get_weights_path(N, forced_rank)
  try:
    print('Loading weights from %s' % weights_path)
    model.load_weights(weights_path)
    msg = 'Existing weights loaded from {}'
  except Exception:
    msg = '{} non existing. No previous weights to be restored.'
    if raise_if_no_weights:
      raise
  print(msg.format(weights_path))
  return weights_path, model
