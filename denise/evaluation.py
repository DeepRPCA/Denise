# Lint as: python3
# pylint: disable=invalid-name
"""Module to evaluate algorithms and dump results to files.

This module does not compare algorithms, nor does it draw any graph/picture.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os.path
import math
import time

from absl import flags
from denise import data
import numpy as np
import tensorflow.compat.v1 as tf

flags.DEFINE_string("results_dir", None,
                    "/path/to/results/dir")
FLAGS = flags.FLAGS


def _get_output_dir(algo_name, N, given_rank, forced_rank, sparsity,
                    l0_available=True, dist=None):
  # Path with one set of results should look like:
  # ${FLAGS.results_dir}/${algo_name}/${size_r0_fr_s0_l0dist}/results_files
  dirname = algo_name.replace(" ", "_").lower()
  if l0_available:
    subdir = "N{}_K{}_fr{}_S0{}".format(N, given_rank, forced_rank, sparsity)
    if dist:
      subdir += "_L0DIST{}".format(dist)
  else:
    subdir = "N{}_market".format(N)
  path = os.path.join(FLAGS.results_dir, dirname, subdir)
  tf.io.gfile.makedirs(path)
  return path


def _run_batch(func, sigma, N, forced_rank, batch_1=False):
  """Runs func for every element of the batch sigma.

  Args:
    batch_1: Bool, defaults to False. when True, run each element independently,
      but in a batch of size 1.
  """
  L = []
  S = []
  durations_s = []
  for sigma_sample in sigma:
    before = time.time()
    res = func(sigma_sample, N, forced_rank)
    after = time.time()
    L_sample, S_sample = res
    durations_s.append(after - before)
    L.append(L_sample)
    S.append(S_sample)
  return durations_s, L, S


def _run_all(func, sigma, N, forced_rank):
  """Runs func over sigma directly."""
  before = time.time()
  L, S = func(sigma, N, forced_rank)
  after = time.time()
  duration_s = after - before
  return [duration_s/len(sigma)] * len(sigma), L, S


def _run(func, M, N, forced_rank, supports_batch,
         preprocessing=None, postprocessing=None, batch_1=False):
  """Runs func over M (aka. sigma)."""
  if preprocessing:
    M, forced_rank = preprocessing(M, forced_rank)
  if supports_batch:
    if batch_1:
      duration_s, L, S = _run_batch(func, M, N, forced_rank, batch_1=batch_1)
    else:
      durations_s, L, S = _run_all(func, M, N, forced_rank)
  else:
    durations_s, L, S = _run_batch(func, M, N, forced_rank)
  if postprocessing:
    L = postprocessing(L)
    S = postprocessing(S)
  return durations_s, L, S


def _save_metrics_to_csv(keys, metrics, fpath):
  """Save metrics (list of tuples) to CSV."""
  with tf.io.gfile.GFile(fpath, "w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(keys)
    for row in metrics:
      writer.writerow(row)


def _get_sparsity(A, tolerance=0.01):
  """Returns ~% of zeros."""
  positives = np.abs(A) > tolerance
  non_zeros = np.count_nonzero(positives)
  return (A.size - non_zeros) / float(A.size)


def run_eval(
    algo_name,
    algo_fct,
    supports_batch,
    dataset,
    N, given_rank, forced_rank, sparsity, rank_tolerance=.01,
    preprocessing=None, postprocessing=None, l0_available=True, batch_1=False,
    dist=None):
  """Runs algo over list of sigma (L0+S0) matrices, dump results to output_dir.

  Files written:
    ${output_dir}/metrics.csv
    ${output_dir}/L.npy
    ${output_dir}/S.npy

  Args:
    algo_name: str, name of the algorithm.
    algo_fct: callable, called as f(sigma, N, forced_rank).
    supports_batch: bool, set to True if sigma can be passed once as an array of
      matrices.
    dataset: dataset containing L0 and S0 matrices to use for evaluation.
      Dimensions must match what model expects, and other params.
    N: int, size of matrices N*N.
    given_rank: int, the rank of original L0 matrice.
    forced_rank: int, for algorithms that need a forced_rank.
    sparsity: int (%), sparsity of S0.
    rank_tolerance:
    preprocessing: function(M, rank) -> M, rank.
    postprocessing: function to apply to L and S.
    l0_available: bool. When True (default), dataset includes {LO, S0}. When
      False, dataset only gives M0.
    batch_1: bool, defaults to False. When True run each element independently,
      but in a batch of size 1.
    dist: None or the used distribution
  """
  output_dir = _get_output_dir(
      algo_name, N, given_rank, forced_rank, sparsity, l0_available, dist)
  if l0_available:
    L0, S0 = data.get_L_S_np_from_dataset(dataset)
    M0 = L0 if S0 is None else L0 + S0
  else:
    M0 = data.get_M_np_from_dataset(dataset)
    S0 = None

  # # To reload the previous results without re-running the algorithms:
  # print("Loading results from %s..." % algo_name)
  # L, S = data.load_L_S(output_dir)
  # durations_s = [1., 3.2]

  print("Evaluating %s..." % algo_name)
  durations_s, L, S = _run(algo_fct, M0, N, forced_rank, supports_batch,
                           preprocessing, postprocessing, batch_1)
  print("Done evaluating %s" % algo_name)
  path_l = os.path.join(output_dir, "L.npy")
  path_s = os.path.join(output_dir, "S.npy")
  print("Saving L and S into %s and %s" % (path_l, path_s))
  data.save_np_arr(L, path_l)
  data.save_np_arr(S, path_s)

  if FLAGS.shrink:
    for S_sample in S:
      S_sample[np.abs(S_sample) < (1 / math.sqrt(20))] = 0.

  print("Computing metrics...")
  rankL = [np.linalg.matrix_rank(L_sample, rank_tolerance) for L_sample in L]
  rankM0 = [np.linalg.matrix_rank(M0_sample, rank_tolerance)
            for M0_sample in M0]
  sparsityS = [_get_sparsity(S_sample) for S_sample in S]

  RE_ML = [np.linalg.norm(M_sample - L_sample) / np.linalg.norm(M_sample)
           for M_sample, L_sample in zip(M0, L)]
  RE_M = [np.linalg.norm(M0_sample - L_sample - S_sample) /
          np.linalg.norm(M0_sample)
          for L_sample, S_sample, M0_sample in zip(L, S, M0)]
  metrics = [
      # metric, mean, std
      ("rankM", np.mean(rankM0), np.std(rankM0)),
      ("rankL", np.mean(rankL), np.std(rankL)),
      ("sparsityS", np.mean(sparsityS), np.std(sparsityS)),
      ("duration_s", np.mean(durations_s), np.std(durations_s)),
      ("RE_ML", np.mean(RE_ML), np.std(RE_ML)),
      ("RE_M", np.mean(RE_M), np.std(RE_M)),
  ]
  if S0 is not None:
    RE_S = [np.linalg.norm(S_sample - S0_sample) / np.linalg.norm(S0_sample)
            for S_sample, S0_sample in zip(S, S0)]
    RE_L = [np.linalg.norm(L_sample - L0_sample) / np.linalg.norm(L0_sample)
            for L_sample, L0_sample in zip(L, L0)]
    metrics.extend([
        ("RE_S", np.mean(RE_S), np.std(RE_S)),
        ("RE_L", np.mean(RE_L), np.std(RE_L)),
    ])

  print(metrics)
  metrics_path = os.path.join(output_dir, "metrics.csv")
  _save_metrics_to_csv(["metric", "mean", "std"], metrics, metrics_path)
  print("Metrics saved to %s" % metrics_path)
