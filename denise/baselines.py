"""Py wrapper to run baseline algorithms.

Users of that library should have matlab installed and should change the path
to the LRS library installed on their machine.
"""
import math
import os

from absl import flags
import matlab
import matlab.engine
import numpy as np

FLAGS = flags.FLAGS

LRS_LIB = os.getenv('LIB_MATLAB_LRS_PATH')
LIBS = ['PROPACK', 'SVD']
ALGOS = ['rpca/PCP','rpca/IALM', 'rpca/FPCP', 'mc/RPCA-GD',
 'mc/PG-RMC','nmf/Semi-NMF', 'nmf/Deep-Semi-NMF', 'rpca/noncvxRPCA']
OTHER_LIBS = [path.strip()
              for path in os.getenv('LIB_MATLAB_MISC', '').split(':')]


def add_path(eng, *components):
  path = os.path.join(LRS_LIB, *components)
  print('Adding path %s' % path)
  eng.addpath(path)


print('Starting matlab engine...')
eng = matlab.engine.start_matlab("-nosoftwareopengl -nodisplay")
for lib in LIBS:
  add_path(eng, 'libs', lib)
for algo in ALGOS:
  add_path(eng, 'algorithms', algo)
for path in OTHER_LIBS:
  eng.addpath(path)

# set number of threads
eng.maxNumCompThreads(1)

# PCP: Robust Principal Component Analysis?
# (Candes et al., PCP, 2009, 46128)
def pcp(M, N, forced_rank):
  lambda_ = 1 / (1.79 * math.sqrt(N))
  if FLAGS.eval_market:
    lambda_ = 1 / (1.56 * math.sqrt(N))
  tol = 1e-5
  return eng.PCP(M, lambda_, tol, nargout=2)


# IALM: The Augmented Lagrange Multiplier Method for Exact Recovery
# of Corrupted Low-Rank Matrices
# (Lin et al., IALM, 2009, 2260)
def ialm(M, N, forced_rank):
  lambda_ = 1 / (1.79 * math.sqrt(N))
  if FLAGS.eval_market:
    lambda_ = 1 / (1.56 * math.sqrt(N))
  return eng.inexact_alm_rpca(M, lambda_, nargout=2)


# FPCP: Fast principal component pursuit via alternating minimization
# (Rodriguez et al., FPCP,  2013 IEEE Int. Conf. on Image Processing, 73)
def fpcp(M, N, forced_rank):
  lambda_ = 1 / math.sqrt(N)
  if FLAGS.shrink:
    return eng.fastpcp(M, lambda_, forced_rank, nargout=2)
  else:
    L, _ = eng.fastpcp(M, lambda_, forced_rank, nargout=2)
    L = np.matrix(L)
    S = M - L
    return L, S


# RPCA-GD: Fast algorithms for robust pca via gradient descent
# (Yi et. al, RPCA-GD, NIPS 2016, 135)
def rpca_gd(M, N, forced_rank):
  U, V = eng.rpca_gd(M, forced_rank, .1, {}, nargout=2)
  U = np.matrix(U)
  V = np.matrix(V)
  L = U.dot(V.getT())
  S = M - L
  return L, S
