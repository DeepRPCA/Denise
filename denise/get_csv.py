

import pandas as pd
import os


algo_dict = {
    "Denise": "dnn_topo0",
    "FPCP": "fpcp",
    "IALM": "ialm",
    "RPCA-GD": "rpcagd",
    "PCP": "pcp",
}

def get_csv(
        N=20, K=3, forced_rank=3, S=(60,70,80,90,95), dist="normal0",
        algos=("PCP", "IALM", "FPCP", "RPCA-GD", "Denise")
):
    data = []
    columns = ["s_S0", "algo", "rankL", "sparsityS", "normLL", "normSS", "time"]
    for s in S:
        for a in algos:
            path = "results/{}/N{}_K{}_fr{}_S0{}_L0DIST{}/metrics.csv".format(
                algo_dict[a], N, K, forced_rank, s, dist)
            df = pd.read_csv(path, index_col=0)
            rank = "{:.2f} ({:.2f})".format(
                df.loc["rankL", "mean"], df.loc["rankL", "std"])
            sparsity = "{:.2f} ({:.2f})".format(
                df.loc["sparsityS", "mean"], df.loc["sparsityS", "std"])
            normL = "{:.2f} ({:.2f})".format(
                df.loc["RE_L", "mean"], df.loc["RE_L", "std"])
            normS = "{:.2f} ({:.2f})".format(
                df.loc["RE_S", "mean"], df.loc["RE_S", "std"])
            time = "{:.2f} ({:.2f})".format(
                df.loc["duration_s", "mean"]*1000,
                df.loc["duration_s", "std"]*1000)
            data.append([s/100, a, rank, sparsity, normL, normS, time])
    df_out = pd.DataFrame(data=data, columns=columns)
    df_out.to_csv("comparisons/comparison_overview_{}.csv".format(dist))


if __name__ == '__main__':
    get_csv(dist="normal0")
    get_csv(dist="student2")


