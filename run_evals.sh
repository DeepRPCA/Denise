RESULTS_DIR="$(pwd)/results"
COMPARISONS_DIR="$(pwd)/comparisons"

if [ "$(hostname)" ==  "ada-17" ]; then
  echo "working on ada-17"
  export LIB_MATLAB_LRS_PATH="/userdata/fkrach/Projects/matlab/lrslibrary"
else
  export LIB_MATLAB_LRS_PATH="/Users/Flo/Code/Matlab/lrslibrary"
fi

# create the directory if it doesn't exist
if [ ! -d "$RESULTS_DIR" ]; then
  mkdir ${RESULTS_DIR}
fi
if [ ! -d "$COMPARISONS_DIR" ]; then
  mkdir ${COMPARISONS_DIR}
fi

# run all the evaluations
for K in 3 #5 7
do
  for s in 60 70 80 90 95
  do
    for dist in normal0 student2
    do
      python denise/script_train_eval.py eval --model=topo0 --weights_dir_path=data/weights/ --N=20 --K=$K --sparsity=$s --results_dir=${RESULTS_DIR} --forced_rank=3 --ldistribution=$dist --shrink=False;
      python denise/script_eval_baselines.py --results_dir=${RESULTS_DIR} --forced_rank=3 --K=$K --N=20 --sparsity=$s --ldistribution=$dist --shrink=False;
      python denise/script_draw_comparison_images.py --N=20 --K=$K --forced_rank=3 --sparsity=$s --results_dir=${RESULTS_DIR} --comparisons_dir=${COMPARISONS_DIR} --ldistribution=$dist --only_table=True;
    done
  done
done


## run all the evaluations
#for K in 5 7
#do
#  for s in 60 70 80 90 95
#  do
#    for dist in normal0 student2
#    do
#      python denise/script_train_eval.py eval --model=topo0 --weights_dir_path=data/weights/ --N=20 --K=$K --sparsity=$s --results_dir=${RESULTS_DIR} --forced_rank=3 --ldistribution=$dist;
#      python denise/script_eval_baselines.py --results_dir=${RESULTS_DIR} --forced_rank=3 --K=$K --N=20 --sparsity=$s --ldistribution=$dist;
#      python denise/script_draw_comparison_images.py --N=20 --K=$K --forced_rank=3 --sparsity=$s --results_dir=${RESULTS_DIR} --comparisons_dir=${COMPARISONS_DIR} --ldistribution=$dist --only_table=True;
#    done
#  done
#done
