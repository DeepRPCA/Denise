"""Various methods to load/save data."""
# Lint as: python3
# pylint: disable=invalid-name
import io
import os.path
import tarfile

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def _get_L(example):
  return example["L"]


def _get_S(example):
  return example["S"]


def get_L_S_np_from_dataset(dataset):
  """Returns np arrays: list of L and S."""
  L = np.array([
      l for l in tfds.as_numpy(dataset.map(_get_L))])
  S = np.array([
      s for s in tfds.as_numpy(dataset.map(_get_S))])
  return L, S


def _get_M(example):
  return example["M"]


def get_M_np_from_dataset(dataset):
  """Returns np arrays: list of L and S."""
  return np.array([
      m for m in tfds.as_numpy(dataset.map(_get_M))])


def save_np_arr(arr, fpath):
  """Save numpy array into fpath."""
  with tf.io.gfile.GFile(fpath, "wb") as f:
    np.save(f, arr, allow_pickle=False)


def load_L_S(dirpath):
  """Returns np arrays for L and S."""
  path_l = os.path.join(dirpath, "L.npy")
  path_s = os.path.join(dirpath, "S.npy")
  with tf.io.gfile.GFile(path_l, "rb") as fileL:
    L = np.load(fileL)
  with tf.io.gfile.GFile(path_s, "rb") as fileS:
    S = np.load(fileS)
  return L, S


class _TarInFile(object):
  """Do not use directly, only by _TarWriter."""

  def __init__(self, name, tar):
    self._name = name
    self._tar = tar
    self._file = io.BytesIO()

  def __enter__(self):
    return self._file

  def __exit__(self, type, value, traceback):
    tarinfo = tarfile.TarInfo(self._name)
    self._file.flush()
    tarinfo.size = self._file.tell()
    self._file.seek(0)
    self._tar.addfile(tarinfo, self._file)


class TarWriter(object):
  def __init__(self, output_path):
    self._file = tf.io.gfile.GFile(output_path, "wb")
    self._tar = tarfile.open(mode="w|", fileobj=self._file)

  def __enter__(self):
    return self

  def __exit__(self, type, value, traceback):
    self._tar.close()

  def add_file(self, name):
    return _TarInFile(name, self._tar)
