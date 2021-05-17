# Lint as: python3
# pylint: disable=invalid-name
"""Train from given data."""
import sys

from absl import app
from absl import flags
from lsr import algo_tf
from lsr import evaluation
from lsr import market_matrices  # pylint: disable=unused-import
from lsr import positive_semidefinite_matrices  # pylint: disable=unused-import
from lsr.script_prepare_datasets import DIR
import tensorflow_datasets as tfds
import tensorflow as tf


FLAGS = flags.FLAGS

flags.DEFINE_integer("N", None, "Size of matrix NxN.")
flags.DEFINE_integer("K", None, "Initial rank of L0.")
flags.DEFINE_integer("forced_rank", None, "Forced rank")
flags.DEFINE_integer("sparsity", None, "Sparsity (%) of S0.")
flags.DEFINE_bool("shrink", True, "Should we shrink while training / eval?")

# Training only:
flags.DEFINE_integer("batch_size", 1024, "Batch size (for training).")
flags.DEFINE_float("learning_rate", 1e-3, "Learing rate (training).")
flags.DEFINE_float(
    "eps_nn", 1e-6,
    "Stops training when diff between 2 loss functions < eps_nn")

# Eval only:
flags.DEFINE_bool(
    "eval_market", None,
    "Set to True to evaluate on market data.")
flags.DEFINE_string(
    "ldistribution", None, "What L distribution to use")
flags.DEFINE_bool(
    "reservoir", None,
    "Reservoir computing: do another learning phase on last layer.")

flags.DEFINE_integer("nb_CPU_inter", 0, "inter op parallelism")
flags.DEFINE_integer("nb_CPU_intra", 0, "intra op parallelism")

def main(argv):
  usage = "Usage: %s ({train,eval}) --data=../data/..." % argv[0]

  # restrict number of CPUs
  if FLAGS.master == 'local':
    tf.config.threading.set_inter_op_parallelism_threads(FLAGS.nb_CPU_inter)
    tf.config.threading.set_intra_op_parallelism_threads(FLAGS.nb_CPU_intra)

  if len(argv) > 2:
    sys.exit(usage)
  action = len(argv) == 2 and argv[1] or "train"

  N, K, forced_rank, sparsity = (
      FLAGS.N, FLAGS.K, FLAGS.forced_rank, FLAGS.sparsity)

  if FLAGS.eval_market:
    if action == "train":
      raise AssertionError("train and --eval_market are incompatible.")
    ds_name = "MarketMatricesTrainVal/N{}".format(N)
    l0_available = False
  else:
    name = {
        "normal": "positive_semidefinite_matrices",
        "normal0": "positive_semidefinite_matrices_l_normal0",
        "normal1": "positive_semidefinite_matrices_l_normal1",
        "normal2": "positive_semidefinite_matrices_l_normal2",
        "normal3": "positive_semidefinite_matrices_l_normal3",
        "normal4": "positive_semidefinite_matrices_l_normal4",
        "uniform": "positive_semidefinite_matrices_l_uniform",
        "uniform1": "positive_semidefinite_matrices_l_uniform1",
        "uniform2": "positive_semidefinite_matrices_l_uniform2",
        "student1": "positive_semidefinite_matrices_l_student1",
        "student2": "positive_semidefinite_matrices_l_student2",
    }[FLAGS.ldistribution]
    ds_name = "{}/N{}_S0.{}_K{}:0.*.*".format(name, N, sparsity, K)
    l0_available = True

  builder = tfds.builder(ds_name, data_dir=DIR)
  builder.download_and_prepare()
  if action == "train":
    print("Training from %s" % ds_name)
    ds = builder.as_dataset(split="train", shuffle_files=True)
    algo_tf.train(
        ds, N, K, forced_rank,
        sparsity, FLAGS.batch_size,
        FLAGS.learning_rate,
        FLAGS.eps_nn,
        builder.info.splits["train"].num_examples,
        shrink=FLAGS.shrink,
        )
  elif action == "eval":
    print("Evaluate %s" % ds_name)
    ds = builder.as_dataset(split="validation")
    lr_nn = algo_tf.LR_NN(N, forced_rank, FLAGS.reservoir)
    algo_name = "dnn_%s" % FLAGS.model
    supports_batch = True
    batch_1 = False
    if FLAGS.reservoir:
      algo_name = algo_name + "_reservoir"
      batch_1 = True
      ds = ds.batch(1)
    evaluation.run_eval(
        algo_name, lr_nn, supports_batch,
        ds, N, K, forced_rank, sparsity, l0_available=l0_available,
        batch_1=batch_1, dist=FLAGS.ldistribution)
  else:
    sys.exit(usage)


if __name__ == "__main__":
  app.run(main)
