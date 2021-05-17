"""Runs baseline algorithms over real-estate input matrix and generates images.

source /home/calypso/work/research/current/denise/code_src/setup_env.sh
export LD_PRELOAD=/usr/local/MATLAB/R2018b/sys/os/glnxa64/libstdc++.so.6.0.22
python3.6 run_realestate.py

"""
from absl import app
from absl import flags

import numpy as np
import math
import os
import baselines
import algo_tf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import matlab
import matlab.engine


FLAGS = flags.FLAGS
flags.DEFINE_bool(
    "eval_market", True,
    "Keep True.")
flags.DEFINE_bool(
    "shrink", False,
    "Set to True to apply shrinkage.")

_ALGOS = [
    ("pcp", baselines.pcp),
    ("ialm", baselines.ialm),
    ("fpcp", baselines.fpcp),
    ("rpcagd", baselines.rpca_gd),
]

perm = np.array(
    [3,25,27,15,16,41,2,12,29,40,14,43,20,21,39,32,10,13,23,24,34,
     31,5,18,33,37,11,42,35,4,7,9,30,26,44,6,19,8,17,36,22,1,28,38]) - 1


def _sublot(matrix, vmin, vmax, output_file):
    plt.imshow(matrix, cmap='Spectral',vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.colorbar()
    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    plt.savefig(output_file)
    plt.close()


def main(argv):
    sigma = np.loadtxt("data/real_estate/input/Sigma_real_estate.csv",
                       delimiter=',')
    sigma = sigma[::2, ::2]
    sigma = sigma[perm, :]
    sigma = sigma[:, perm]
    sigma = sigma[:20, :20]
    N = 20
    k = 3
    vmin, vmax = -0.5, 0.5

    if not os.path.exists("../latex_src/images/real_estate/"):
        os.makedirs("../latex_src/images/real_estate/")

    # denise
    sigma_b = np.expand_dims(sigma, axis=0)
    algo = "dnn_topo0"
    lr_nn = algo_tf.LR_NN(N, k, False)
    L, S = lr_nn(sigma_b, N, k)
    L = L[0]
    S = S[0]
    if FLAGS.shrink:
        S[np.abs(S) < (1 / math.sqrt(20))] = 0.

    rank_l = np.linalg.matrix_rank(L, 0.01)
    print(algo, "rank(L)=", rank_l)
    print(algo, "reconstruction error: {}".format(
        np.linalg.norm(sigma-L-S, ord='fro')))
    _sublot(L, vmin, vmax,
            "../latex_src/images/real_estate/L_%s.pdf" % algo)
    _sublot(S, vmin, vmax,
            "../latex_src/images/real_estate/S_%s.pdf" % algo)

    # baselines
    sigma = matlab.double(sigma.tolist())
    for algo, algo_fct in _ALGOS:
        L, S = algo_fct(sigma, N, float(k))
        L = np.matrix(L)
        S = np.matrix(S)
        if FLAGS.shrink:
            S[np.abs(S) < (1 / math.sqrt(20))] = 0.
        rank_l = np.linalg.matrix_rank(L, 0.01)
        print(algo, "rank(L)=", rank_l)
        print(algo, "reconstruction error: {}".format(
            np.linalg.norm(sigma-L-S, ord='fro')))
        _sublot(L, vmin, vmax,
                "../latex_src/images/real_estate/L_%s.pdf" % algo)
        _sublot(S, vmin, vmax,
                "../latex_src/images/real_estate/S_%s.pdf" % algo)


if __name__ == "__main__":
    app.run(main)
