#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
NPROC=$(($(echo $CUDA_VISIBLE_DEVICES | grep -o "," | wc -l)+1))

SRC_ROOT=$HOME/work/llama2.cnn_dailymail.eval/src/my_scripts

source $SRC_ROOT/start_here.sh

for MODEL_NAME in sparse_ft@SRCcerebras50@lr1e-4@WD0.0@B8@GrAcc8@W0.1@ep2@GPUs7@ID15577
do
    M=$HOME/models/llama2/cnn_dailymail/llama-recipes/sparse_finetuned/$MODEL_NAME
    accelerate launch --config_file $SRC_ROOT/accelerate_default_config.${NPROC}gpus.yaml $SRC_ROOT/rouge_accelerate.py --model-path $M --batch 2 --samples 16 --generation top_k --top-k 2 --max-new-tokens 100 --first-k-preds 3 --use-accelerate 1 --output-dir rouge
done

