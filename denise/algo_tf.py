"""Defines the training procedure and loss functions for Denise."""
import tempfile

from absl import flags
import tensorflow as tf
from tensorflow import keras as k

from lsr import model_lib_tf

FLAGS = flags.FLAGS

flags.DEFINE_string('tb_logdir', None, 'Path for tensorboard logs.')

flags.DEFINE_string('master', None, 'TPU worker.')
flags.DEFINE_integer(
    'num_cores', None,
    ('Number of TPU cores. Used to compute the correct global batch size.'
     '8 for v2-8, 32 for v2-32, ...'))

flags.DEFINE_string('loss', 'L1', 'Loss function.')
flags.DEFINE_boolean('use_s', True, 'Set to False to have Sigma = L.')


def _get_tb_logdir():
  """Returns path to directory where to write tensor board logs."""
  if FLAGS.tb_logdir:
    return FLAGS.tb_logdir
  return tempfile.mkdtemp()


class LR_NN(object):

  SHRINK = False

  def __init__(self, N, forced_rank, reservoir=False):
    self._N = N
    self._forced_rank = forced_rank
    self._reservoir = reservoir
    self._init_model()

  def _init_model(self):
    _, self._model = model_lib_tf.get_model(
        self._N, self._forced_rank, shrink=self.SHRINK,
        raise_if_no_weights=True,
        freeze_internal_layers=self._reservoir)

  def __call__(self, sigma_np, N, forced_rank):
    """Run model on batch of sigma, returns (L, S)."""
    n = batch_size = len(sigma_np)
    if self._reservoir:
      # reservoir is not used by Denise. This is an experimentation to see if
      # an additional iteration of gradient descent on evaluated data improves
      # the quality of the result. Answer: it does not seem to do so
      # significantly, yet degrades the time performance a lot.
      given_rank, sparsity, eps_nn = None, None, None  # Unused
      learning_rate = 0.001
      sigma_ds = tf.data.Dataset.from_tensor_slices(sigma_np)
      train(sigma_ds, N, given_rank, forced_rank, sparsity, batch_size,
            learning_rate, eps_nn, n, model=self._model, epochs=1,
            reservoir=True)
    S, L = self._model.predict(sigma_np, batch_size=batch_size)
    return L, S


class LR_NN_shrink(LR_NN):

  # Denise applies shrinkage, similarily to FPCP.
  SHRINK = True


def train(
    training_ds,
    N,
    given_rank,
    forced_rank,
    sparsity,
    batch_size,
    learning_rate,
    eps_nn,
    n,
    shrink=False,
    model=None,
    epochs=100,
    reservoir=False,
    ):
  """
  Args:
    training_ds: The dataset containing L0 and S0. The model is fed M0=L0+S0 and
      is expected to return L and S. Loss uses S and S0.
    N: Matrices are of size N*N.
    given_rank: The original rank of L0. Unused.
    forced_rank: The forced rank the model should be trained with.
    sparsity: The original sparsity of S0. Unused.
    batch_size: How many matrices should be used for each training iteration.
    learning_rate: Optimizer learning rate.
    eps_nn: Stops training when diff between 2 loss functions < eps_nn.
    n: number of matrices in the dataset.
    shrink: bool. Whether or not the model should apply shrinkage (like FCPC).
    model: optional. Keras model to use (locally), if given.
    epochs: The number of epoch to train.
    reservoir: Defaults to False. When True, train is not really a training, and
      won't save any weights, nor export anything to tensorboard.
  """

  del given_rank
  del sparsity
  # Give sigma for training and S0 and L0 for loss functions.
  def transform(rec):
    S, L = rec['S'], rec['L']
    sigma = S + L
    if not FLAGS.use_s:
      return L, (S, L)
    return sigma, (S, L)
  def transform_reservoir(rec):
    # rec = sigma
    # L and S are not used anyway, so just return sigma instead, as it has the
    # same dimensions.
    return rec, (rec, rec)
  transform = reservoir and transform_reservoir or transform
  training_ds = training_ds.map(transform).repeat()

  def loss_S(S0, S):
    if FLAGS.loss == 'L1':
      return k.backend.sum(k.backend.abs(S))
    elif FLAGS.loss == 'frobenus':
      return tf.keras.backend.sqrt(
          k.backend.sum(k.backend.pow(S, 2)))
    elif FLAGS.loss == 'L1_S_diff':
        return k.backend.sum(k.backend.abs(S - S0))
    else:
      raise AssertionError('Unknown specified norm: %s' % FLAGS.loss)

  def loss_L(L0, L):
    del L0  # Unused
    del L
    return tf.constant(0.)

  print('Use TPU at %s' % (
      FLAGS.master if FLAGS.master is not None else 'local'))
  if FLAGS.loss == 'binary_crossentropy':
    loss = 'binary_crossentropy'
  else:
    loss=[loss_S, loss_L]
  if model:
    my_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
    model.compile(optimizer=my_optimizer, loss=loss, metrics=['accuracy'])
  elif FLAGS.master == 'local':
    weights_path, model = model_lib_tf.get_model(N, forced_rank, shrink=shrink)
    my_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
    model.compile(optimizer=my_optimizer, loss=loss, metrics=['accuracy'])
  else:
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=FLAGS.master)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.experimental.TPUStrategy(resolver)

    with strategy.scope():
      weights_path, model = model_lib_tf.get_model(N, forced_rank, shrink=shrink)

      my_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
      my_optimizer = tf.tpu.CrossShardOptimizer(my_optimizer)

      model.compile(optimizer=my_optimizer,
                    loss=loss,
                    metrics=['accuracy'])

  if reservoir:
    callbacks = []
  else:
    tboard_logs = _get_tb_logdir()
    print("TensorBoard logs at %s" % tboard_logs)
    # tboard = subprocess.Popen('tensorboard --logdir=%s' % tboard_logs, shell=True)
    callbacks = [
        k.callbacks.EarlyStopping(monitor='loss', min_delta=eps_nn,
                                  patience=5000),
        k.callbacks.ModelCheckpoint(
            weights_path, monitor='loss', save_weights_only=True,
            save_best_only=True),
        k.callbacks.TensorBoard(log_dir=tboard_logs, batch_size=batch_size),
    ]

  training_ds = training_ds.batch(batch_size)
  print("training dataset", training_ds)
  steps_per_epoch = n // batch_size
  print('steps per epoch:', steps_per_epoch)
  model.fit(training_ds,
            epochs=epochs,
            callbacks=callbacks,
            steps_per_epoch=steps_per_epoch,
           )
