# Lint as: python3
# pylint: disable=invalid-name
"""Script to evaluate matlab baseline algorithms."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from denise import baselines
from denise import evaluation
from denise import positive_semidefinite_matrices  # pylint: disable=unused-import
from denise import market_matrices  # pylint: disable=unused-import
from denise.script_prepare_datasets import DIR
import matlab
import numpy as np
import tensorflow_datasets as tfds

FLAGS = flags.FLAGS

flags.DEFINE_integer("N", None, "Size of matrix NxN.")
flags.DEFINE_integer("K", None, "Initial rank of L0.")
flags.DEFINE_integer("forced_rank", None, "Forced rank")
flags.DEFINE_integer("sparsity", None, "Sparsity (%) of S0.")
flags.DEFINE_bool(
    "eval_market", None,
    "Set to True to evaluate on market data.")
flags.DEFINE_string(
    "ldistribution", None, "What L distribution to use")
flags.DEFINE_bool("shrink", True, "Should we shrink while training / eval?")

_ALGOS = [
    ("pcp", baselines.pcp),
    ("ialm", baselines.ialm),
    ("fpcp", baselines.fpcp),
    ("rpcagd", baselines.rpca_gd),
]


def _eval_baselines(N, sparsity, K, forced_rank):
  if FLAGS.eval_market:
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
  print("Loading validation dataset %s" % ds_name)
  dataset = tfds.load(ds_name, split="validation", data_dir=DIR)
  preprocessing = lambda M, fr: (matlab.double(M.tolist()), float(fr))
  postprocessing = lambda m_list: [np.matrix(m) for m in m_list]
  for algo_name, fct in _ALGOS:
    evaluation.run_eval(
        algo_name, fct, False,  # supports batch
        dataset, N, K, forced_rank, sparsity,
        preprocessing=preprocessing, postprocessing=postprocessing,
        l0_available=l0_available, dist=FLAGS.ldistribution)


def main(argv):
  del argv
  N, K, forced_rank, sparsity = (
      FLAGS.N, FLAGS.K, FLAGS.forced_rank, FLAGS.sparsity)
  _eval_baselines(N, sparsity, K, forced_rank)


if __name__ == "__main__":
  app.run(main)
