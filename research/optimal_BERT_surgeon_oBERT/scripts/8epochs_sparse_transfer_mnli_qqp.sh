#!/bin/bash

# TASK can be either mnli or qqp
export TASK=mnli
# export TASK=qqp

# 90% upstream pruned oBERT model: neuralmagic/oBERT-12-upstream-pruned-unstructured-90-v2
# 97% upstream pruned oBERT model: neuralmagic/oBERT-12-upstream-pruned-unstructured-97-v2
export UPSTREAM_PRUNED_MODEL=neuralmagic/oBERT-12-upstream-pruned-unstructured-90-v2

export RECIPE=research/optimal_BERT_surgeon_oBERT/recipes/8epochs_sparse_transfer_${TASK}.yaml

CUDA_VISIBLE_DEVICES=0 python src/sparseml/transformers/text_classification.py \
  --distill_teacher neuralmagic/oBERT-teacher-${TASK} \
  --model_name_or_path ${UPSTREAM_PRUNED_MODEL} \
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
  --learning_rate 1.5e-4 \
  --max_seq_length 128 \
  --preprocessing_num_workers 8 \
  --seed 42 \
  --num_train_epochs 8 \
  --recipe ${RECIPE} \
  --output_dir transformers_output_dir \
  --overwrite_output_dir \
  --skip_memory_metrics true \
  --report_to wandb