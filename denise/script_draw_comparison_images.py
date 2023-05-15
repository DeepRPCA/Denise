# Lint as: python3
# pylint: disable=invalid-name
"""Draw image comparing pictures of decompositions from various methods."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os.path
import math

from absl import app
from absl import flags
from denise import data
from denise import positive_semidefinite_matrices  # pylint: disable=unused-import
from denise.script_prepare_datasets import DIR
import matplotlib
matplotlib.use("Agg")
# pylint: disable=g-import-not-at-top
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds

try:
    from telegram_notifications import send_bot_message as SBM
except Exception:
    class SendBotMessage:
        def __init__(self):
            pass

        @staticmethod
        def send_notification(text, *args, **kwargs):
            print(text)

    SBM = SendBotMessage()

FLAGS = flags.FLAGS

flags.DEFINE_string("results_dir", None,
                    "/path/to/results/dir")
flags.DEFINE_string("comparisons_dir", None,
                    "/path/to/comparisons/dir")
flags.DEFINE_integer("N", None, "Size of matrix NxN.")
flags.DEFINE_integer("K", None, "Initial rank of L0.")
flags.DEFINE_integer("forced_rank", None, "Forced rank")
flags.DEFINE_integer("sparsity", None, "Sparsity (%) of S0.")
flags.DEFINE_bool("shrink", True, "Should we shrink while training / eval?")

flags.DEFINE_boolean("only_table", False, "When True, only generate PDF table.")

flags.DEFINE_string(
    "ldistribution", None, "What L distribution to use")

flags.DEFINE_bool(
    "eval_market", None,
    "Set to True to evaluate on market data.")

# Maps from display name to directory name in results dir.
_ALGOS = [
    ("PCP", "pcp"),
    ("IALM", "ialm"),
    ("FPCP", "fpcp"),
    ("RPCA-GD", "rpcagd"),
    ("Denise", "dnn_topo0"),
]


class PlotComparisons(object):
  """Helper to draw pictures comparing results of various algorithms."""

  _NCOLS = 3  # Sigma, S, L
  _FONTSIZE = 20

  def __init__(self, L0, S0):
    """L0 and S0 are batches of L0 and S0 matrices."""
    self.L0 = L0
    self.S0 = S0
    self._algos = collections.OrderedDict()  # {"algo_name": (L, S)}
    if L0 is not None:
      self.add_algo("Original", L0, S0)

  def add_algo(self, algo_name, L, S):
    self._algos[algo_name] = (L, S)

  def _start_plotting_comparison(self, i):
    """Starts a comparison figure."""
    self._group_size = len(self._algos)
    if self.L0 is not None:
      self._group_size += 1
    comparison_fig, comparison_axes = plt.subplots(
        nrows=self._group_size,
        ncols=self._NCOLS,
        figsize=(4 * self._group_size, self._NCOLS * 4))
    comparison_axes[0, 0].set_title("Sigma", fontsize=self._FONTSIZE)
    comparison_axes[0, 1].set_title("L", fontsize=self._FONTSIZE)
    comparison_axes[0, 2].set_title("S", fontsize=self._FONTSIZE)
    if self.L0 is not None:
      sigma0 = self.L0[i] + self.S0[i]
      comparison_axes[0, 0].set_ylabel("Original", fontsize=self._FONTSIZE)
      comparison_axes[0, 0].imshow(sigma0, cmap="Spectral", vmin=-1, vmax=1,
                                   interpolation="none")
      comparison_axes[0, 1].imshow(self.L0[i], cmap="Spectral", vmin=-1, vmax=1,
                                   interpolation="none")
      comparison_axes[0, 2].imshow(self.S0[i], cmap="Spectral", vmin=-1, vmax=1,
                                   interpolation="none")
    return comparison_fig, comparison_axes

  def _plot_comparison(self, output_f, idx, results):
    """Plots and save one image, comparing Sigma, L and S from various algos.

    Args:
      output_f: file-like object in which to write the picture.
      idx: indice of matrice being compared.
      results: list of tuples (algo, L, S)).
    """
    comparison_fig, comparison_axes = self._start_plotting_comparison(idx)

    start = 0 if self.L0 is None else 1
    vmin = -100
    vmax = 100
    for i, (algo, L, S) in enumerate(results, start=start):
      comparison_axes[i, 0].imshow(L+S, cmap="Spectral", vmin=vmin, vmax=vmax,
                                   interpolation="none")
      comparison_axes[i, 0].set_ylabel(algo, fontsize=self._FONTSIZE)
      comparison_axes[i, 1].imshow(L, cmap="Spectral", vmin=vmin, vmax=vmax,
                                   interpolation="none")
      comparison_axes[i, 2].imshow(S, cmap="Spectral", vmin=vmin, vmax=vmax,
                                   interpolation="none")

    # Remove ticks + print comparison table.
    for i in range(self._group_size):
      for j in range(self._NCOLS):
        comparison_axes[i, j].set_xticklabels([])
        comparison_axes[i, j].set_yticklabels([])

    comparison_fig.subplots_adjust(hspace=.2, wspace=.2)
    plt.savefig(output_f, transparent=True, bbox_inches='tight',
                pad_inches = 0)
    plt.close()

  def export_images(self, output_path, max_number_examples=10):
    """Export max_examples images to output_path."""
    with data.TarWriter(output_path) as tar:
      for i in range(max_number_examples):
        results = [(algo, L[i], S[i])
                   for (algo, (L, S)) in self._algos.items()]
        with tar.add_file("%s.png" % i) as fileobj:
          self._plot_comparison(fileobj, i, results)
        for algo, L, S in results:
          for label, matrix in [('L', L), ('S', S), ('M', S+L)]:
            with tar.add_file("%d/%s_%s.png" % (i, algo, label)) as fileobj:
              fig, ax = plt.subplots()
              im = ax.imshow(matrix, cmap="Spectral", vmin=-1, vmax=1,
                        interpolation="none")
              ax.axis('off')  # clear x-axis and y-axis
              fig.colorbar(im, ax=ax)
              plt.tight_layout(pad=0, w_pad=0, h_pad=0)
              plt.savefig(fileobj,transparent = True,
                          bbox_inches = 'tight', pad_inches = 0)
              plt.close(fig)


def draw_comparison_images(N, K, forced_rank, sparsity, l0_available, dist=None):
  """Draws the comparison images and save them into a tar file."""
  if l0_available:
    result_subdir = "N{}_K{}_fr{}_S0{}".format(N, K, forced_rank, sparsity)
    if dist:
      result_subdir += "_L0DIST{}".format(dist)
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
      }[dist]
      ds_name = "{}/N{}_S0.{}_K{}:0.*.*".format(name, N, sparsity, K)
    else:
      ds_name = "positive_semidefinite_matrices/N{}_S0.{}_K{}:0.*.*".format(
        N, sparsity, K)
    print("Loading validation dataset %s" % ds_name)
    dataset = tfds.load(ds_name, split="validation", data_dir=DIR)
    L0, S0 = data.get_L_S_np_from_dataset(dataset)
  else:
    result_subdir = "N{}_market".format(N)
    L0 = S0 = None
  print("Looking for available results in %s/*/%s" % (
      FLAGS.results_dir, result_subdir))

  plot_comparisons = PlotComparisons(L0, S0)
  for algo_name, dirname in _ALGOS:
    print("Loading L and S for %s" % algo_name)
    algo_results_path = os.path.join(FLAGS.results_dir, dirname, result_subdir)
    if not tf.io.gfile.exists(algo_results_path):
      print("\t %s not found, skipping." % algo_results_path)
      continue
    L, S = data.load_L_S(algo_results_path)
    # shrink matrices
    if FLAGS.shrink:
        indices_to_shrink = np.abs(S) < (1 / math.sqrt(20))
        S[indices_to_shrink] = 0
    plot_comparisons.add_algo(algo_name, L, S)

  tf.io.gfile.makedirs(FLAGS.comparisons_dir)
  comparison_path = os.path.join(
      FLAGS.comparisons_dir, "%s.tar" % result_subdir)
  print("Writing comparison images to %s" % comparison_path)
  plot_comparisons.export_images(comparison_path)


_PDF_METRIC_CELL = r"{:.2f} ({:.2f})"
_PDF_METRIC_CELL_T = r"{:.2f} ({:.2f})"
# "duration_s" is hardcoded
_PDF_METRICS = ["rankL", "sparsityS"]
_PDF_METRICS_L0_AVAILABLE = ["RE_L", "RE_S"]  # , "RE_M"]
_PDF_METRICS_L0_UNVAILABLE = ["RE_ML", "RE_M"]
_PDF_METRICS_H = "Method & rank & sparsity &"
_PDF_METRICS_L0_AVAILABLE_H = r"""
$RE_{L} =\frac{||L-L_0||_F}{||L_0||_F}$ &
$RE_{S} =\frac{||S-S_0||_F}{||S_0||_F}$ & """
# $RE_{M} =\frac{||M-L-S||_F}{||M||_F}$ &
_PDF_METRICS_L0_UNVAILABLE_H = r"""
$RE_{ML} = \frac{||M-L||_F}{||M||_F}$ &
$RE_{M} = \frac{||M-L-S||_F}{||M||_F}$ & """
_PDF_ROW_TMPL = r"{} & {}\\"
_PDF_TABLE_TMPL = r"""
\begin{table*}[t!]
\label{%(label)s}
\vskip 0.15in
\begin{center}
\begin{small}
\begin{sc}
\begin{tabular}{lccccccr}
\toprule
%(headers)s
Time $(ms)$  \\
\midrule
%(cells)s
\bottomrule
\end{tabular}
\end{sc}
\end{small}
\end{center}
\vskip -0.1in
\caption{%(caption)s
}
\end{table*}
"""


def generate_pdf_table(N, K, forced_rank, sparsity, l0_available, dist=None):
  """Generate PDF table with all numeric results."""
  if l0_available:
    result_subdir = "N{}_K{}_fr{}_S0{}".format(N, K, forced_rank, sparsity)
    if dist:
      result_subdir += "_L0DIST{}".format(dist)
  else:
    result_subdir = "N{}_market".format(N)
  table_path = os.path.join(
      FLAGS.comparisons_dir, "%s.tex" % result_subdir)
  print("Generating comparison table in %s" % table_path)
  rows = []
  for method, dirname in _ALGOS:
    metrics_path = os.path.join(FLAGS.results_dir, dirname, result_subdir,
                                "metrics.csv")
    print("Reading metrics from %s in %s" % (method, metrics_path))
    with tf.io.gfile.GFile(metrics_path, "r") as metricsf:
      df = pd.read_csv(metricsf).set_index("metric")
    if l0_available:
      pdf_metrics = _PDF_METRICS + _PDF_METRICS_L0_AVAILABLE
      headers = _PDF_METRICS_H + _PDF_METRICS_L0_AVAILABLE_H
    else:
      pdf_metrics = _PDF_METRICS + _PDF_METRICS_L0_UNVAILABLE
      headers = _PDF_METRICS_H + _PDF_METRICS_L0_UNVAILABLE_H
    cells = [_PDF_METRIC_CELL.format(df["mean"][metric], df["std"][metric])
             for metric in pdf_metrics]
    cells.append(_PDF_METRIC_CELL_T.format(
        df["mean"]["duration_s"] * 1E3, df["std"]["duration_s"]))
    if method == "Denise":
      method = r"\textbf{%s }" % method
      cells = [r"\textbf{%s }" % cell for cell in cells]
    row = _PDF_ROW_TMPL.format(method, " & ".join(cells))
    rows.append(row)
  if l0_available:
    caption = ("Comparison of the evaluation metrics: average and standard"
               " deviation over the full synthetic evaluation dataset.")
    label = "synthetictable"
  else:
    caption = ("Comparison of the evaluation metrics: average and standard"
               " deviation over the full real dataset.")
    label = "realtable"
  pdf_table = _PDF_TABLE_TMPL % {"cells": "\n".join(rows), "headers": headers,
                                 "caption": caption, "label": label}
  with tf.io.gfile.GFile(table_path, "w") as tablef:
    tablef.write(pdf_table)
  print("done")


def main(argv):
  del argv
  N, K, forced_rank, sparsity, dist = (
      FLAGS.N, FLAGS.K, FLAGS.forced_rank, FLAGS.sparsity, FLAGS.ldistribution)
  l0_available = not FLAGS.eval_market
  generate_pdf_table(N, K, forced_rank, sparsity, l0_available, dist)
  SBM.send_notification(
      text="finished evaluation: {}, {}".format(sparsity, dist)
  )
  if FLAGS.only_table:
    return
  draw_comparison_images(N, K, forced_rank, sparsity, l0_available, dist)


if __name__ == "__main__":
  app.run(main)
