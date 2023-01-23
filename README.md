# Denise

[![DOI](https://zenodo.org/badge/367672358.svg)](https://zenodo.org/badge/latestdoi/367672358)

## Installation

To install the necessary python dependencies:

```sh
# cd into the same directory as this README file.

# Install virtualenv, use equivalent command for non debian based systems.
sudo apt-get install python3-venv

python3 -m venv py3
# or virtualenv --python=/usr/bin/python3.6 py3
source py3/bin/activate

pip install --upgrade pip
pip install tensorflow
pip install pandas

pip3.8 install -e .
```

## Data download and extraction

```sh
wget https://deeprpca.s3.eu-central-1.amazonaws.com/data.zip
unzip data.zip
```

## read dataset
```sh
python3 lsr/script_read_datasets.py
```


## Datasets preparation

**NOTE**: The following is not needed if Denise is trained/evaluated first (since it
automatically generates datasets which do not exist).

```sh
python3 lsr/script_prepare_datasets.py
```

If you do intend to run the training, you will also need the synthetic dataset,
which is about 31GB for matrices of size 20x20.

```sh
python3 lsr/script_prepare_datasets.py --synthetic=true
```

To only generate the synthetic evaluation data:

```sh
python3 lsr/script_prepare_datasets.py --market=false --synthetic_validation=true
```

## Training of Denise locally

This step is optional, you can pass and directly use the weights from
the data directory.

Training locally:

```sh
python lsr/script_train_eval.py train \
  --batch_size=512 \
  --N=20 \
  --K=3 \
  --forced_rank=3 \
  --sparsity=95 \
  --ldistribution=normal \
  --model=topo0 \
  --master=local \
  --weights_dir_path=data/weights2 \
  --loss={L1, L1_S_diff}
  --shrink=False \
  --nb_CPU_inter=0 --nb_CPU_intra=0
```

## Retraining Denise on Market data

```sh
python lsr/retrain_market.py --weights_dir_path=data/weights2_retrain --trained_weights_dir_path=data/weights2 --N=20 --K=3 --forced_rank=3 --sparsity=95 --model=topo0 --shrink=False --epochs=100 --batch_size=58
```


## Training on Google cloud using TPUs

Create your own TPU cloud instance by following the [official documentation](https://cloud.google.com/tpu/docs/quickstart).

For quick reference, at the time of writing this, run the following command in cloud shell:

```
ctpu up --name host-v2-8 --zone europe-west4-a --tpu-size v2-8
```

SSH to the VM instance created for your TPU above in step one above.
Install dependencies as described earlier, using Google storage to store prepare data.
Run training as describe at previous step, using `--master` flag to specify TPU to be used.

## Evaluation of Denise

On synthetic dataset:

```sh
RESULTS_DIR="$(pwd)/results"
mkdir ${RESULTS_DIR}
```

```sh
python lsr/script_train_eval.py eval \
  --model=topo0 \
  --weights_dir_path=data/weights/ \
  --N=20 \
  --K=3 \
  --sparsity={60,70,80,90,95} \
  --results_dir=${RESULTS_DIR} \
  --forced_rank=3 \
  --shrink={False,True} \
  --ldistribution={normal, uniform, normal0, normal1, normal2, normal3, normal4, uniform1, uniform2, student1, student2}
```

or run all the evaluations on different synthetic datasets (i.e. for all
combinations of sparsity and ldistribution) also for baselines:
```sh
./run_evals.sh
```

On real dataset:

```sh
python lsr/script_train_eval.py eval \
  --eval_market \
  --model=topo0 \
  --weights_dir_path=data/weights2/ \
  --N=20 \
  --K=3 \
  --sparsity=95 \
  --results_dir=${RESULTS_DIR} \
  --shrink={False, True} \
  --forced_rank=3
```

### Evaluation of baselines

You'll need Matlab and the Python API. See
https://ch.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html
for instructions.

Install the needed Matlab library: https://github.com/andrewssobral/lrslibrary
On OSX some files of the library do not work and have to be changed.

Shell script `setup_env.sh` file sets the environment so following commands can
be run. Edit the file to set the correct values and run:

```sh
source lsr/setup_env.sh
```

or:
```sh
export LIB_MATLAB_LRS_PATH="yourpath"
```

On synthetic dataset:

```sh
python3 lsr/script_eval_baselines.py \
  --results_dir=${RESULTS_DIR} \
  --forced_rank=3 \
  --K=3 \
  --N=20 \
  --shrink={False, True} \
  --sparsity=95
```

Using the synthetic eval data generated with other distributions:

```sh
python3 lsr/script_eval_baselines.py \
  --results_dir=${RESULTS_DIR} \
  --forced_rank=3 \
  --K=3 \
  --N=20 \
  --sparsity={60,70,80,90,95} \
  --shrink={False, True} \
  --ldistribution={normal, uniform, normal1, normal2, normal3, normal4, uniform1, uniform2, student1, student2}
```



On real dataset:

```sh
python3 lsr/script_eval_baselines.py \
  --eval_market \
  --results_dir=${RESULTS_DIR} \
  --forced_rank=3 \
  --shrink={False, True} \
  --N=20
```

## Generation of pictures and comparison tables

```sh
COMPARISONS_DIR="$(pwd)/comparisons"
mkdir ${COMPARISONS_DIR}
```

On synthetic dataset results:

```sh
python3 lsr/script_draw_comparison_images.py \
  --N=20 \
  --K=3 \
  --forced_rank=3 \
  --sparsity={60,70,80,90,95} \
  --results_dir=${RESULTS_DIR} \
  --comparisons_dir=${COMPARISONS_DIR}
  --ldistribution={normal, uniform, normal1, normal2, normal3, normal4, uniform1, uniform2, student1, student2}
```

On market dataset results:

```sh
python3 lsr/script_draw_comparison_images.py \
  --eval_market \
  --N=20 \
  --forced_rank=3 \
  --results_dir=${RESULTS_DIR} \
  --shrink={False, True} \
  --comparisons_dir=${COMPARISONS_DIR}
```

## Generation of joint table
cd into code_src (and adjust the following two files as needed), then:

```sh
python lsr/get_csv.py; python lsr/create_tex_table.py
```


## Eval on real estate dataset
cd into code_src:

```sh
python lsr/run_realestate.py --model=topo0 --weights_dir_path=data/weights2/ --shrink=True
```


## Computing the Frobenius norm of `M - L - S`

The list of algos is read from `script_draw_comparison_images.py`.
The evaluation must has been run before hand, has matrices are read
from results folder.

```sh
python3 lsr/script_m_minus_l_minus_s.py --results_dir=$RESULTS_DIR
```

## License

This code can be used in accordance with the LICENSE.

Citation
--------

If you this library useful, please cite our paper:
[Denise: Deep Robust PCA for Positive Semidefinite Matrices](https://arxiv.org/abs/2004.13612).
```
@article{OptStopRandNN2021,
author    = {Herrera, Calypso and Krach, Florian and Kratsios, Anastasis and Ruyssen, Pierre and Teichmann, Josef },
title     = {Denise: Deep Robust PCA for Positive Semidefinite Matrices},
journal   = {CoRR},
volume    = {abs/2104.13669},
year      = {2020},
url       = {https://arxiv.org/abs/2004.13612}}
```

Last Page Update: **22/05/2021**
