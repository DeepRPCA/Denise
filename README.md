# Denise

[![DOI](https://zenodo.org/badge/367672358.svg)](https://zenodo.org/badge/latestdoi/367672358)

This repository contains the code for the paper 
[Denise: Deep Robust Principal Component Analysis for Positive Semidefinite Matrices](https://openreview.net/forum?id=D45gGvUZp2).


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

pip install -e .
```

## Data download and extraction

The pretrained weights for the experiments shown in the paper are available 
for download via the following command:

```sh
wget https://polybox.ethz.ch/index.php/s/ESx1cuJEboxikRB/download -O data.zip
unzip data.zip
rm data.zip
```

This creates a directory `data/` with the trained weights in `data/weights/` and 
the retrained weights (for real world dataset) in `data/weights_retrain/`.

If these weights should be used for evaluation, replace the `weights_dir_path` in the [evaluation section](#evaluation-of-denise)
by `data/weights/` or `data/weights_retrain/` respectively.


There are also (old) pretrained weights available, which were trained only via 
the unsupervised loss function. Download them via the following command:
```sh
wget https://deeprpca.s3.eu-central-1.amazonaws.com/data.zip
unzip data.zip
```


## Datasets preparation

**NOTE**: The following is not needed if Denise is trained/evaluated first (since it
automatically generates datasets which do not exist).


The following command creates the Market dataset and prepares it for training
```sh
python3 denise/script_prepare_datasets.py --market=true
```

If you do intend to run the training, you will also need the synthetic dataset,
which is about 31GB for matrices of size 20x20.

```sh
python3 denise/script_prepare_datasets.py --synthetic=true
```

To only generate the synthetic evaluation data:

```sh
python3 denise/script_prepare_datasets.py --market=false --synthetic_validation=true
```

## Training
### Training of Denise locally

This step is optional, you can pass and directly use the downloaded weights.
The training will save the trained weights in `weights_dir_path`, which is
`data/weights2/` by default. To use these weights for evaluation, replace the
`weights_dir_path` in the [evaluation section](#evaluation-of-denise) by
`data/weights2/`.

Training locally:

```sh
python denise/script_train_eval.py train \
  --batch_size=512 \
  --N=20 \
  --K=3 \
  --nb_epochs=90 \
  --forced_rank=3 \
  --sparsity=95 \
  --ldistribution=normal \
  --model=topo0 \
  --master=local \
  --weights_dir_path=data/weights2 \
  --loss={L1, L1_S_diff} \
  --shrink=False \
  --nb_CPU_inter=0 --nb_CPU_intra=0
```

### Retraining Denise on Market data

The following command retrains Denise on Market data using the weights from 
`trained_weights_dir_path` as starting point and stores the retrained weights in 
`weights_dir_path`.

```sh
```sh
python denise/retrain_market.py --weights_dir_path=data/weights2_retrain --trained_weights_dir_path=data/weights2 --N=20 --K=3 --forced_rank=3 --sparsity=95 --model=topo0 --shrink=False --epochs=100 --batch_size=58
```


### Training on Google cloud using TPUs

Create your own TPU cloud instance by following the [official documentation](https://cloud.google.com/tpu/docs/quickstart).

For quick reference, at the time of writing this, run the following command in cloud shell:

```
ctpu up --name host-v2-8 --zone europe-west4-a --tpu-size v2-8
```

SSH to the VM instance created for your TPU above in step one above.
Install dependencies as described earlier, using Google storage to store prepare data.
Run training as describe at previous step, using `--master` flag to specify TPU to be used.


## Evaluation
### Evaluation of Denise

On synthetic dataset:

```sh
RESULTS_DIR="$(pwd)/results"
mkdir ${RESULTS_DIR}
```

```sh
python denise/script_train_eval.py eval \
  --model=topo0 \
  --weights_dir_path=data/weights/ \
  --N=20 \
  --K=3 \
  --sparsity={60,70,80,90,95} \
  --results_dir=${RESULTS_DIR} \
  --forced_rank=3 \
  --shrink=False \
  --ldistribution={normal, uniform, normal0, normal1, normal2, normal3, normal4, uniform1, uniform2, student1, student2}
```

or run all the evaluations on different synthetic datasets (i.e. for all 
combinations of sparsity and ldistribution) also for baselines:
```sh
./run_evals.sh
```

On real dataset:

```sh
python denise/script_train_eval.py eval \
  --eval_market \
  --model=topo0 \
  --weights_dir_path=data/weights/ \
  --N=20 \
  --K=3 \
  --sparsity=95 \
  --results_dir=${RESULTS_DIR} \
  --forced_rank=3 \
  --shrink=False
  
```

### Evaluation of baselines

You'll need Matlab and the Python API. See
https://ch.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html
for instructions.
We used Matlab version R2018b which is compatible with python 3.6.

Install the needed Matlab library: https://github.com/andrewssobral/lrslibrary
On OSX some files of the library do not work and have to be changed.

as described above, all baselines can be evaluated (together with Denise) by running:
```sh
./run_evals.sh
```

otherwise, run the following command with the correct path to the installed lrs matlab library:

```sh
export LIB_MATLAB_LRS_PATH="/Users/Flo/Code/Matlab/lrslibrary"
# export LIB_MATLAB_LRS_PATH="/userdata/fkrach/Projects/matlab/lrslibrary"
```

On synthetic dataset:

```sh
python3 denise/script_eval_baselines.py \
  --results_dir=${RESULTS_DIR} \
  --forced_rank=3 \
  --K=3 \
  --N=20 \
  --sparsity=95 \
  --ldistribution=normal \
  --shrink=False 
```

Using the synthetic eval data generated with other distributions:

```sh
python3 denise/script_eval_baselines.py \
  --results_dir=${RESULTS_DIR} \
  --forced_rank=3 \
  --K=3 \
  --N=20 \
  --sparsity={60,70,80,90,95} \
  --shrink=False \
  --ldistribution={normal, uniform, normal1, normal2, normal3, normal4, uniform1, uniform2, student1, student2}
```



On real world (stock market) dataset:

```sh
python3 denise/script_eval_baselines.py \
  --eval_market \
  --results_dir=${RESULTS_DIR} \
  --forced_rank=3 \
  --shrink=False \
  --N=20
```



### Generation of pictures and comparison tables

```sh
COMPARISONS_DIR="$(pwd)/comparisons"
mkdir ${COMPARISONS_DIR}
```

On synthetic dataset results:

```sh
python3 denise/script_draw_comparison_images.py \
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
python3 denise/script_draw_comparison_images.py \
  --eval_market \
  --N=20 \
  --forced_rank=3 \
  --results_dir=${RESULTS_DIR} \
  --shrink=False \
  --comparisons_dir=${COMPARISONS_DIR}
```

### Generation of joint table
cd into code_src (and adjust the following two files as needed), then:

```sh
python denise/get_csv.py; python denise/create_tex_table.py
```


### Eval on real estate dataset
cd into code_src:

```sh
python denise/run_realestate.py --model=topo0 --weights_dir_path=data/weights/ --shrink=False
```


### Computing the Frobenius norm of `M - L - S`

The list of algos is read from `script_draw_comparison_images.py`.
The evaluation must have been run beforehand, since matrices are read
from results folder.

```sh
python3 denise/script_m_minus_l_minus_s.py --results_dir=$RESULTS_DIR
```



--------------------------------------------------------------------------------
## Usage, License & Citation

This code can be used in accordance with the [LICENSE](LICENSE).

If you find this code useful or include parts of it in your own work, 
please cite our paper:  

[Denise: Deep Robust Principal Component Analysis for Positive Semidefinite Matrices](https://openreview.net/forum?id=D45gGvUZp2)

```
@article{
herrera2023denise,
title={Denise: Deep Robust Principal Component Analysis for Positive Semidefinite Matrices},
author={Calypso Herrera and Florian Krach and Anastasis Kratsios and Pierre Ruyssen and Josef Teichmann},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2023},
url={https://openreview.net/forum?id=D45gGvUZp2},
note={}
}
```
