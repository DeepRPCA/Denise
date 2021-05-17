# Lint as: python3
# pylint: disable=invalid-name
"""Positive semidefinite matrices, used as sythetic dataset to train Denise."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow_datasets.public_api as tfds

_CITATION = """"""
_DESCRIPTION = """Low + sparse matrix decomposition dataset."""


_TRAIN_SIZE = int(10*1E6)  # 10M
_EVAL_SIZE = int(1E4)  # 10K


def _get_sparsity(matrix, tolerance=0.01):
  """Returns ~% of zeros."""
  cnt = len(matrix.flatten())
  non_zeros = np.count_nonzero(np.abs(matrix) > tolerance)
  return (cnt-non_zeros) / cnt


def _get_sparse_matrix(size_n, given_sparsity):
  """Returns matrix of size_n*size_n, with given_sparsity (max)."""
  sparse = np.zeros((size_n, size_n), dtype=np.float32)
  current_sparsity = 1.0
  while current_sparsity > given_sparsity:
    j = np.random.randint(1, size_n)
    i = np.random.randint(0, j)
    b = np.random.uniform(-1, 1)
    a = np.random.uniform(abs(b), 1)
    sparse[i, i] = sparse[j, j] = a
    sparse[i, j] = sparse[j, i] = b
    current_sparsity = _get_sparsity(sparse)
  return sparse


def _get_sparse_matrix_diag1(size_n, given_sparsity, l):
    """
    Returns matrix of size_n*size_n, with given_sparsity (max).
    It chooses S (and scales L) s.t. the final matrix has 1's on the diag.
    """
    sparse = _get_sparse_matrix(size_n, given_sparsity)
    m = l + sparse
    m_max = np.max(np.abs(m))
    l = l/m_max
    sparse = sparse / m_max
    m_diag = np.diag(l + sparse)
    s_add = 1 - m_diag
    sparse = sparse + np.diag(s_add)
    return sparse, l




def _get_lr_matrix_normal(size_n, K):
  """Returns low-rank matrix of size size_n*size_n, of rank K.

  Uses statandard normal distribution.
  """
  a = np.random.randn(size_n, K).astype(np.float32)
  low_rank = np.matmul(a, a.transpose((1, 0)))
  return low_rank


def _get_lr_matrix_normal1(size_n, K):
  """Returns low-rank matrix of size size_n*size_n, of rank K.

  Uses statandard normal distribution.
  """
  a = np.random.normal(loc=1, scale=1., size=(size_n, K)).astype(np.float32)
  low_rank = np.matmul(a, a.transpose((1, 0)))
  return low_rank


def _get_lr_matrix_normal2(size_n, K):
  """Returns low-rank matrix of size size_n*size_n, of rank K.

  Uses statandard normal distribution.
  """
  a = np.random.normal(loc=0, scale=2., size=(size_n, K)).astype(np.float32)
  low_rank = np.matmul(a, a.transpose((1, 0)))
  return low_rank


def _get_lr_matrix_normal3(size_n, K):
  """Returns low-rank matrix of size size_n*size_n, of rank K.

  Uses statandard normal distribution.
  """
  a = np.random.normal(loc=0, scale=0.5, size=(size_n, K)).astype(np.float32)
  low_rank = np.matmul(a, a.transpose((1, 0)))
  return low_rank


def _get_lr_matrix_normal4(size_n, K):
  """Returns low-rank matrix of size size_n*size_n, of rank K.

  Uses statandard normal distribution.
  """
  a = np.random.normal(loc=-1., scale=1., size=(size_n, K)).astype(np.float32)
  low_rank = np.matmul(a, a.transpose((1, 0)))
  return low_rank


def _get_lr_matrix_studt1(size_n, K):
  """Returns low-rank matrix of size size_n*size_n, of rank K.

  Uses student t distribution.
  """
  a = np.random.standard_t(df=10, size=(size_n, K)).astype(np.float32)
  low_rank = np.matmul(a, a.transpose((1, 0)))
  return low_rank


def _get_lr_matrix_studt2(size_n, K):
  """Returns low-rank matrix of size size_n*size_n, of rank K.

  Uses student t distribution.
  """
  a = np.random.standard_t(df=5, size=(size_n, K)).astype(np.float32)
  low_rank = np.matmul(a, a.transpose((1, 0)))
  return low_rank


def _get_lr_matrix_uniform(size_n, K):
  """Returns low-rank matrix of size size_n*size_n, of rank K.

  Uses uniform distribution.
  """
  a = np.random.uniform(size=(size_n, K)).astype(np.float32)
  low_rank = np.matmul(a, a.transpose((1, 0)))
  return low_rank


def _get_lr_matrix_uniform1(size_n, K):
  """Returns low-rank matrix of size size_n*size_n, of rank K.

  Uses uniform distribution.
  """
  a = np.random.uniform(low=-1., high=1., size=(size_n, K)).astype(np.float32)
  low_rank = np.matmul(a, a.transpose((1, 0)))
  return low_rank


def _get_lr_matrix_uniform2(size_n, K):
  """Returns low-rank matrix of size size_n*size_n, of rank K.

  Uses uniform distribution.
  """
  a = np.random.uniform(low=-1., high=0., size=(size_n, K)).astype(np.float32)
  low_rank = np.matmul(a, a.transpose((1, 0)))
  return low_rank


class PositiveSemidefiniteMatricesConfig(tfds.core.BuilderConfig):
  """BuilderConfig for randomly generate positive semidefinite matrices."""

  def __init__(self, size_n, sparsity, rank):
    """Constructs a PositiveSemidefiniteMatricesConfig.

    Args:
      size_n: size of the matrices (n*n).
      sparsity: float, the sparsity of the sparse matrice.
      rank: K, the maximum rank of the low_rank matrice.
    """
    name = "N%s_S%.2f_K%s" % (size_n, sparsity, rank)
    description = "%s*%s matrices, sparsity=%.2f, max rank=%s" % (
        size_n, size_n, sparsity, rank)
    super(PositiveSemidefiniteMatricesConfig, self).__init__(
        name=name, version=tfds.core.Version("0.1.0"), description=description)
    self.size_n = size_n
    self.sparsity = sparsity
    self.rank = rank


class PositiveSemidefiniteMatrices(tfds.core.GeneratorBasedBuilder):
  """Builder for randomly generate positive semidefinite matrices."""
  DISTRIBUTION_L = "normal"

  BUILDER_CONFIGS = [
      PositiveSemidefiniteMatricesConfig(size_n=10, sparsity=0.95, rank=2),

      PositiveSemidefiniteMatricesConfig(size_n=20, sparsity=0.95, rank=3),
      PositiveSemidefiniteMatricesConfig(size_n=20, sparsity=0.9, rank=3),
      PositiveSemidefiniteMatricesConfig(size_n=20, sparsity=0.8, rank=3),
      PositiveSemidefiniteMatricesConfig(size_n=20, sparsity=0.7, rank=3),
      PositiveSemidefiniteMatricesConfig(size_n=20, sparsity=0.6, rank=3),

      PositiveSemidefiniteMatricesConfig(size_n=20, sparsity=0.95, rank=5),
      PositiveSemidefiniteMatricesConfig(size_n=20, sparsity=0.9, rank=5),
      PositiveSemidefiniteMatricesConfig(size_n=20, sparsity=0.8, rank=5),
      PositiveSemidefiniteMatricesConfig(size_n=20, sparsity=0.7, rank=5),
      PositiveSemidefiniteMatricesConfig(size_n=20, sparsity=0.6, rank=5),

      PositiveSemidefiniteMatricesConfig(size_n=20, sparsity=0.95, rank=7),
      PositiveSemidefiniteMatricesConfig(size_n=20, sparsity=0.9, rank=7),
      PositiveSemidefiniteMatricesConfig(size_n=20, sparsity=0.8, rank=7),
      PositiveSemidefiniteMatricesConfig(size_n=20, sparsity=0.7, rank=7),
      PositiveSemidefiniteMatricesConfig(size_n=20, sparsity=0.6, rank=7),

      PositiveSemidefiniteMatricesConfig(size_n=40, sparsity=0.95, rank=3),
      PositiveSemidefiniteMatricesConfig(size_n=80, sparsity=0.95, rank=3),
  ]

  def _info(self):
    n = self.builder_config.size_n
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            "L": tfds.features.Tensor(shape=(n, n), dtype=tf.float32),
            "S": tfds.features.Tensor(shape=(n, n), dtype=tf.float32),
        }),
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs={"seed": 4242, "num_matrices": _TRAIN_SIZE},
        ),
        tfds.core.SplitGenerator(
            name=tfds.Split.VALIDATION,
            gen_kwargs={"seed": 4243, "num_matrices": _EVAL_SIZE},
        ),
    ]

  def _get_lr_matrix(self, *args, **kwargs):
    return {
      "normal": _get_lr_matrix_normal,
      "normal1": _get_lr_matrix_normal1,
      "normal2": _get_lr_matrix_normal2,
      "normal3": _get_lr_matrix_normal3,
      "normal4": _get_lr_matrix_normal4,
      "uniform": _get_lr_matrix_uniform,
      "uniform1": _get_lr_matrix_uniform1,
      "uniform2": _get_lr_matrix_uniform2,
      "student1": _get_lr_matrix_studt1,
      "student2": _get_lr_matrix_studt2,
    }[self.DISTRIBUTION_L](*args, **kwargs)

  def _generate_examples(self, seed, num_matrices):
    """Yields examples."""
    np.random.seed(seed)
    for i in range(num_matrices):
      l = self._get_lr_matrix(self.builder_config.size_n,
                              self.builder_config.rank)
      s = _get_sparse_matrix(self.builder_config.size_n,
                             self.builder_config.sparsity)
      # s, l = _get_sparse_matrix_diag1(
      #     self.builder_config.size_n, self.builder_config.sparsity, l)
      # print("L", l)
      # print("S", s)
      # print("M", l+s)
      # print("sparsity S (true): {}\nsparsity S (0.22 tol): {}".format(
      #     _get_sparsity(s, tolerance=0), _get_sparsity(s, tolerance=0.22)))
      # raise AssertionError
      yield i, {"L": l, "S": s}


class PositiveSemidefiniteMatricesLUniform(PositiveSemidefiniteMatrices):
  """Same as PositiveSemidefiniteMatrices, but using Uniform distribution for L.

  This dataset only generates a validation split.
  """
  DISTRIBUTION_L = "uniform"

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.VALIDATION,
            gen_kwargs={"seed": 4244, "num_matrices": _EVAL_SIZE},
        ),
    ]


class PositiveSemidefiniteMatricesLUniform1(PositiveSemidefiniteMatrices):
  """Same as PositiveSemidefiniteMatrices, but using Uniform distribution for L.

  This dataset only generates a validation split.
  """
  DISTRIBUTION_L = "uniform1"

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.VALIDATION,
            gen_kwargs={"seed": 4244, "num_matrices": _EVAL_SIZE},
        ),
    ]


class PositiveSemidefiniteMatricesLUniform2(PositiveSemidefiniteMatrices):
  """Same as PositiveSemidefiniteMatrices, but using Uniform distribution for L.

  This dataset only generates a validation split.
  """
  DISTRIBUTION_L = "uniform2"

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.VALIDATION,
            gen_kwargs={"seed": 4244, "num_matrices": _EVAL_SIZE},
        ),
    ]


class PositiveSemidefiniteMatricesLNormal0(PositiveSemidefiniteMatrices):
  """Same as PositiveSemidefiniteMatrices, but using Uniform distribution for L.

  This dataset only generates a validation split.
  """
  DISTRIBUTION_L = "normal"

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.VALIDATION,
            gen_kwargs={"seed": 4244, "num_matrices": _EVAL_SIZE},
        ),
    ]


class PositiveSemidefiniteMatricesLNormal1(PositiveSemidefiniteMatrices):
  """Same as PositiveSemidefiniteMatrices, but using Uniform distribution for L.

  This dataset only generates a validation split.
  """
  DISTRIBUTION_L = "normal1"

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.VALIDATION,
            gen_kwargs={"seed": 4244, "num_matrices": _EVAL_SIZE},
        ),
    ]


class PositiveSemidefiniteMatricesLNormal2(PositiveSemidefiniteMatrices):
  """Same as PositiveSemidefiniteMatrices, but using Uniform distribution for L.

  This dataset only generates a validation split.
  """
  DISTRIBUTION_L = "normal2"

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.VALIDATION,
            gen_kwargs={"seed": 4244, "num_matrices": _EVAL_SIZE},
        ),
    ]


class PositiveSemidefiniteMatricesLNormal3(PositiveSemidefiniteMatrices):
  """Same as PositiveSemidefiniteMatrices, but using Uniform distribution for L.

  This dataset only generates a validation split.
  """
  DISTRIBUTION_L = "normal3"

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.VALIDATION,
            gen_kwargs={"seed": 4244, "num_matrices": _EVAL_SIZE},
        ),
    ]


class PositiveSemidefiniteMatricesLNormal4(PositiveSemidefiniteMatrices):
  """Same as PositiveSemidefiniteMatrices, but using Uniform distribution for L.

  This dataset only generates a validation split.
  """
  DISTRIBUTION_L = "normal4"

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.VALIDATION,
            gen_kwargs={"seed": 4244, "num_matrices": _EVAL_SIZE},
        ),
    ]


class PositiveSemidefiniteMatricesLStudent1(PositiveSemidefiniteMatrices):
  """Same as PositiveSemidefiniteMatrices, but using Uniform distribution for L.

  This dataset only generates a validation split.
  """
  DISTRIBUTION_L = "student1"

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.VALIDATION,
            gen_kwargs={"seed": 4244, "num_matrices": _EVAL_SIZE},
        ),
    ]


class PositiveSemidefiniteMatricesLStudent2(PositiveSemidefiniteMatrices):
  """Same as PositiveSemidefiniteMatrices, but using Uniform distribution for L.

  This dataset only generates a validation split.
  """
  DISTRIBUTION_L = "student2"

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.VALIDATION,
            gen_kwargs={"seed": 4244, "num_matrices": _EVAL_SIZE},
        ),
    ]
