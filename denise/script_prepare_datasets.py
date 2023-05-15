"""Script to prepare the synthetic and real datasets."""
import tensorflow_datasets as tfds

import socket, os

from absl import app
from absl import flags
from denise import market_matrices  # py-lint: disable=unused-import
from denise import positive_semidefinite_matrices  # py-lint: disable=unused-import

DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '../data'))


flags.DEFINE_bool("market", False, "Generate market dataset.")

flags.DEFINE_bool(
    "synthetic", False,
    "Set to True to also generate synthetic dataset (~31GB).")

flags.DEFINE_bool(
    "synthetic_validation", False,
    "Set to True to also generate synthetic validation dataset.")

def main(argv):
  del argv

  if flags.FLAGS.market:
    market_builder = tfds.builder("market_matrices/N20", data_dir=DIR)
    market_builder.download_and_prepare()

  if flags.FLAGS.synthetic:
    synthetic_builder = tfds.builder(
        "positive_semidefinite_matrices/N20_S0.95_K3", data_dir=DIR)
    synthetic_builder.download_and_prepare()

  if flags.FLAGS.synthetic_validation:
    # builder = tfds.builder(
    #     "PositiveSemidefiniteMatricesLUniform/N20_S0.95_K3", data_dir=DIR)
    # builder.download_and_prepare()

    for s in [0.6, 0.7, 0.8, 0.9, 0.95]:
      for dist in ['Normal0', 'Normal1',
                   'Normal2', 'Normal3', 'Normal4',
                   'Uniform', 'Uniform1', 'Uniform2',
                   ]:
        builder = tfds.builder(
          "PositiveSemidefiniteMatricesL{}/N20_S{:.2f}_K3".format(
            dist, s), data_dir=DIR)
        builder.download_and_prepare()


if __name__ == "__main__":
  app.run(main)
