"""The following script computes M - L - S on market dataset.

The evaluation must has been run before hand, has matrices are read
from results folder.
"""
import os.path

from absl import flags
from absl import app
import numpy as np
import tensorflow_datasets as tfds

from denise import market_matrices  # pylint: disable=unused-import
from denise import script_draw_comparison_images

FLAGS = flags.FLAGS

ALGOS = script_draw_comparison_images._ALGOS


def compute_diff(M_list, dir_path, algo, ds_name):
  S_list = np.load(os.path.join(dir_path, algo, ds_name, "S.npy"))
  L_list = np.load(os.path.join(dir_path, algo, ds_name, "L.npy"))
  norms = []
  for M, S, L in zip(M_list, S_list, L_list):
    diff = M - S - L
    norm = np.linalg.norm(diff, ord='fro')
    norms.append(norm)
  mean_norm = np.mean(norms)
  std = np.std(norms)
  return "{:.2f}$\pm${:.2f}".format(mean_norm, std)


def main(argv):
  del argv
  market_ds = tfds.load("market_matrices/N20", split="validation")
  M_list = [e['M'] for e in tfds.as_numpy(market_ds)]
  for algo_nice, algo in ALGOS:
    res = compute_diff(M_list, FLAGS.results_dir, algo, "N20_market")
    print(algo_nice, res, "\n")


if __name__ == "__main__":
  app.run(main)
