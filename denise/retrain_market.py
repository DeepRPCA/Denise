"""Retrain already trained models on market data"""
import sys
import shutil

from absl import app
from absl import flags
from lsr import algo_tf
from lsr import evaluation
from lsr import market_matrices  # pylint: disable=unused-import
from lsr import positive_semidefinite_matrices  # pylint: disable=unused-import
from lsr import model_lib_tf
from lsr import evaluation
from lsr.script_prepare_datasets import DIR
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow import keras as k
import numpy as np


FLAGS = flags.FLAGS

flags.DEFINE_integer("N", None, "Size of matrix NxN.")
flags.DEFINE_integer("K", None, "Initial rank of L0.")
flags.DEFINE_integer("forced_rank", None, "Forced rank")
flags.DEFINE_integer("sparsity", None, "Sparsity (%) of S0.")
flags.DEFINE_bool("shrink", False, "Should we shrink while training / eval?")

# Training only:
flags.DEFINE_integer("batch_size", 100, "Batch size (for training).")
flags.DEFINE_float("learning_rate", 1e-3, "Learing rate (training).")
flags.DEFINE_float(
    "eps_nn", 1e-6,
    "Stops training when diff between 2 loss functions < eps_nn")

flags.DEFINE_string(
    'trained_weights_dir_path', None,
    'Path to directory where to load weights from.')
flags.DEFINE_integer("epochs", 10, "number epochs to retrain")



def retrain(argv):
    # copy weights
    shutil.copytree(FLAGS.trained_weights_dir_path, FLAGS.weights_dir_path)

    # get dataset
    ds_name = "MarketMatricesTrainVal/N{}".format(FLAGS.N)
    builder = tfds.builder(ds_name, data_dir=DIR)
    builder.download_and_prepare()
    ds_train = builder.as_dataset(split="train", shuffle_files=True)
    ds_val = builder.as_dataset(split="validation")

    def transform(rec):
        M = rec['M']
        return M, (M, M)
    training_ds = ds_train.map(transform).repeat()
    val_ds = ds_val.map(transform).repeat()

    # losses and metrics
    def loss_S(S0, S):
        return k.backend.sum(k.backend.abs(S))

    def loss_L(L0, L):
        del L0  # Unused
        del L
        return tf.constant(0.)

    def sparsity(A, tolerance=0.01):
        """Returns ~% of zeros."""
        positives = tf.math.abs(A) > tolerance
        non_zeros = tf.cast(tf.math.count_nonzero(positives), tf.float32)
        size_A = tf.cast(tf.size(A), tf.float32)
        return (size_A - non_zeros) / size_A

    def sparsity_metric(y_true, y_pred):
        sparsityS = []
        for i in range(FLAGS.batch_size):
            sparsityS.append(sparsity(y_pred[i]))
        return tf.math.reduce_mean(sparsityS)

    def ML_metric(y_true, y_pred):
        RE_ML = []
        for i in range(FLAGS.batch_size):
            M_sample, L_sample = y_true[i], y_pred[i]
            RE_ML.append(
                tf.norm(M_sample - L_sample) / tf.norm(M_sample))
        return tf.math.reduce_mean(RE_ML)

    # load model
    weights_path, model = model_lib_tf.get_model(
        FLAGS.N, FLAGS.forced_rank, shrink=False)
    my_optimizer = tf.compat.v1.train.AdamOptimizer(FLAGS.learning_rate)
    model.compile(optimizer=my_optimizer, loss=[loss_S, loss_L],
                  metrics=[[sparsity_metric], [ML_metric]])

    # training
    tboard_logs = algo_tf._get_tb_logdir()
    print("TensorBoard logs at %s" % tboard_logs)
    # tboard = subprocess.Popen('tensorboard --logdir=%s' % tboard_logs, shell=True)
    callbacks = [
        k.callbacks.EarlyStopping(monitor='loss', min_delta=FLAGS.eps_nn,
                                  patience=5000),
        k.callbacks.ModelCheckpoint(
            weights_path, monitor='loss', save_weights_only=True,
            save_best_only=True),
        k.callbacks.TensorBoard(log_dir=tboard_logs, batch_size=FLAGS.batch_size),
    ]
    training_ds = training_ds.batch(FLAGS.batch_size)
    print("training dataset", training_ds)
    n = builder.info.splits["train"].num_examples
    steps_per_epoch = n // FLAGS.batch_size
    print('size:', n, 'steps per epoch:', steps_per_epoch)

    n_val = builder.info.splits["validation"].num_examples
    steps_per_epoch_val = n_val // FLAGS.batch_size
    val_ds = val_ds.batch(FLAGS.batch_size)
    print("validation dataset", val_ds)
    print('size:', n_val, 'steps per epoch:', steps_per_epoch_val)

    # eval before retraining
    eval_losses = model.evaluate(x=val_ds, steps=steps_per_epoch_val)
    loss_names = model.metrics_names
    print("before retraining:")
    print(loss_names)
    print(eval_losses)

    # retrain
    history = model.fit(
        x=training_ds, epochs=FLAGS.epochs, callbacks=callbacks,
        steps_per_epoch=steps_per_epoch, validation_data=val_ds,
        validation_steps=steps_per_epoch_val)
    
    history = history.history
    ind = np.argmin(history["L_ML_metric"])
    print("minimal L_ML_metric (on train set): {}".format(history["L_ML_metric"][ind]))
    print("validation errors at minimal training ML-reconstruction loss:")
    print("val_S_sparsity_metric: {}, val_L_ML_metric: {}".format(
        history["val_S_sparsity_metric"][ind], history["val_L_ML_metric"][ind]))
    
    ind = np.argmax(history["S_sparsity_metric"])
    print("max S_sparsity_metric (on train set): {}".format(history["S_sparsity_metric"][ind]))
    print("validation errors at max training S_sparsity_metric:")
    print("val_S_sparsity_metric: {}, val_L_ML_metric: {}".format(
        history["val_S_sparsity_metric"][ind], history["val_L_ML_metric"][ind]))



if __name__ == '__main__':
    app.run(retrain)
