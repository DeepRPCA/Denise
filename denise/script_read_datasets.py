"""Script to read and check if the synthetic is okay."""
import tensorflow_datasets as tfds

from absl import app
from absl import flags
from lsr import market_matrices  # py-lint: disable=unused-import
from lsr import positive_semidefinite_matrices  # py-lint: disable=unused-import

flags.DEFINE_bool(
    "synthetic", False,
    "Set to True to also generate synthetic dataset (~31GB).")


def main(argv):
  del argv


  synthetic_builder = tfds.builder(
      "positive_semidefinite_matrices/N20_S0.95_K3")
  synthetic_builder.download_and_prepare()

  ds = synthetic_builder.as_dataset(split="train", shuffle_files=True)
  for example in ds:
    print("example")
    print


if __name__ == "__main__":
  app.run(main)
