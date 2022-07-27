#!/bin/bash

export RECIPE=research/optimal_BERT_surgeon_oBERT/recipes/MY_COOL_RECIPE_NAME.yaml
# TASK can be either mnli or qqp
export TASK=mnli
# export TASK=qqp

CUDA_VISIBLE_DEVICES=0 python src/sparseml/transformers/text_classification.py \
  --distill_teacher neuralmagic/oBERT-teacher-${TASK} \
  --model_name_or_path bert-base-uncased \
  --task_name ${TASK} \
  --do_train \
  --fp16 \
  --do_eval \
  --optim adamw_torch \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --save_total_limit 1 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --learning_rate 5e-5 \
  --max_seq_length 128 \
  --preprocessing_num_workers 8 \
  --seed 42 \
  --num_train_epochs 30 \
  --recipe ${RECIPE} \
  --output_dir transformers_output_dir \
  --overwrite_output_dir \
  --skip_memory_metrics true \
  --report_to wandb