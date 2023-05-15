"""TFDS dataset with SP500 stock return covariance matrices."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import itertools
import os.path

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets.public_api as tfds


_CITATION = """https://openreview.net/forum?id=D45gGvUZp2"""
_DESCRIPTION = """Market correlation matrices dataset."""

_SIZE = 1000

_DATA_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '../data'))

_GICS_SECTOR_GROUPS = [
    "Energy", "Materials", "Industrials", "Real Estate",
    "Consumer Discretionary", "Consumer Staples", "Health Care", "Financials",
    "Information Technology", "Communication Services", "Utilities"
]


def _get_symbols_to_watch(symbols, max_stocks_wanted):
  # Create a data frame containing all the Sectors.
  df_comp = pd.read_csv(os.path.join(_DATA_DIR, "comp.csv"))
  # We count how many sectors are ine the global matrix
  d = {"Symbol": symbols}
  # We construct a data frame that contains all the symbols
  df_sub_comp = pd.DataFrame(data=d)
  # For each symbol, we write the Sector and the sub sector
  for symb in symbols:
    index = df_sub_comp.index[df_sub_comp["Symbol"] == symb].tolist()[0]
    cond = df_comp["Symbol"] == symb
    df_sub_comp.at[index,
                   "GICS Sector"] = df_comp[cond]["GICS Sector"].tolist()[0]
    df_sub_comp.at[
        index,
        "GICS Sub Industry"] = df_comp[cond]["GICS Sub Industry"].tolist()[0]
    # we sort the DataFrame, firt by sectors and then by subsectors
    df_sub_comp.sort_values(
        by=["GICS Sector", "GICS Sub Industry"], inplace=True)
  symbols_to_watch = []
  for group in _GICS_SECTOR_GROUPS:
    idx = df_sub_comp["GICS Sector"] == group
    max_not_reached = len(symbols_to_watch) <= max_stocks_wanted
    cond = idx & max_not_reached
    symbols_to_watch.extend(list(df_sub_comp[cond]["Symbol"]))
  symbols_to_watch = symbols_to_watch[:max_stocks_wanted]
  return symbols_to_watch


def _get_sp_matrices(max_stocks_wanted, date_begin="1998-01-01",
                     date_end="2019-01-01"):
  df = pd.read_csv(os.path.join(_DATA_DIR, "sp500.csv"))
  cond_date_inf = df["date"] > date_begin
  cond_date_sup = df["date"] < date_end
  nb_obs = max_stocks_wanted * 5
  delta_step = 5
  # Construct a data frame with all the quotation in the time interval.
  df_int = df[cond_date_inf & cond_date_sup]
  # Drop all columns where quotation are missing in this interval.
  df_int_in = df_int.dropna(axis="columns")

  # Construct a list with all the symbols that remains in this data frame.
  symbols = list(df_int_in.columns[1:])
  symbols_to_watch = _get_symbols_to_watch(symbols, max_stocks_wanted)

  filtered_prices = pd.DataFrame(
      collections.OrderedDict(
          (col, df_int_in[col]) for col in symbols_to_watch))
  filtered_prices.dropna(inplace=True)  #  in case there is a zero or a N/A
  indexes = list(filtered_prices.index.values)
  total_nb_obs = filtered_prices.shape[0]
  total_nb_stocks = filtered_prices.shape[1]
  nb_corr = int((total_nb_obs - nb_obs) / delta_step)

  # Construct the correlation matrix with chosen Stocks and others paremeters.
  for i in range(nb_obs, total_nb_obs, delta_step):
    sub_indexes = indexes[i:i + nb_obs]
    sub_prices_df = filtered_prices.loc[sub_indexes]
    # We compute the returns of the matrix with the prices
    returns_df = sub_prices_df.shift(-1) / sub_prices_df - 1.
    returns_df.dropna(inplace=True)
    # We compute the correlation matrix of the matrix of returns
    sub_corr_df = returns_df.corr()
    sub_corr_matrix = sub_corr_df.to_numpy()
    yield np.float32(sub_corr_matrix)


class MarketConfig(tfds.core.BuilderConfig):

  def __init__(self, size_n):
    name = "N%s" % size_n
    description = "%s*%s correlation matrices from market." % (
        size_n, size_n)
    super(MarketConfig, self).__init__(
        name=name, version=tfds.core.Version("0.1.0"), description=description)
    self.size_n = size_n


class MarketMatrices(tfds.core.GeneratorBasedBuilder):
  """Builder for SP500 stock returns covariance matrices."""

  BUILDER_CONFIGS = [
      MarketConfig(size_n=20),
  ]

  def _info(self):
    n = self.builder_config.size_n
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            "M": tfds.features.Tensor(shape=(n, n), dtype=tf.float32),
        }),
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.VALIDATION,
            gen_kwargs={},
        ),
    ]

  def _generate_examples(self):
    """Yields examples."""
    matrices = _get_sp_matrices(max_stocks_wanted=self.builder_config.size_n)
    for i, matrix in enumerate(itertools.islice(matrices, _SIZE)):
      yield i, {"M": matrix}


class MarketMatricesTrainVal(tfds.core.GeneratorBasedBuilder):
    """Builder for SP500 stock returns covariance matrices."""

    BUILDER_CONFIGS = [
        MarketConfig(size_n=20),
    ]

    def _info(self):
        n = self.builder_config.size_n
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                "M": tfds.features.Tensor(shape=(n, n), dtype=tf.float32),
            }),
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={"date_begin": "1998-01-01", "date_end": "2014-01-01"},
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.VALIDATION,
                gen_kwargs={"date_begin": "2014-01-01", "date_end": "2019-01-01"},
            ),
        ]

    def _generate_examples(self, date_begin, date_end):
        """Yields examples."""
        matrices = _get_sp_matrices(
            max_stocks_wanted=self.builder_config.size_n, date_begin=date_begin,
            date_end=date_end)
        for i, matrix in enumerate(itertools.islice(matrices, 0, None)):
            yield i, {"M": matrix}
