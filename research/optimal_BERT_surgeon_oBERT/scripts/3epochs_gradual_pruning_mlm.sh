#!/bin/bash

# 90% recipe: 3epochs_unstructured90_mlm.yaml
# 97% recipe: 3epochs_unstructured97_mlm.yaml
export RECIPE=research/optimal_BERT_surgeon_oBERT/recipes/3epochs_unstructured90_mlm.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 src/sparseml/transformers/masked_language_modeling.py \
  --distill_teacher neuralmagic/oBERT-12-upstream-pretrained-dense \
  --model_name_or_path neuralmagic/oBERT-12-upstream-pretrained-dense \
  --dataset_name bookcorpus \
  --dataset_name_2 wikipedia \
  --dataset_config_name_2 20200501.en \
  --do_train \
  --fp16 \
  --do_eval \
  --optim adamw_torch \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --save_total_limit 1 \
  --logging_strategy steps \
  --logging_steps 100 \
  --per_device_train_batch_size 32 \
  --gradient_accumulation_steps 2 \
  --per_device_eval_batch_size 16 \
  --learning_rate 5e-4 \
  --weight_decay 0.01 \
  --num_train_epochs 3 \
  --max_seq_length 512 \
  --seed 42 \
  --recipe ${RECIPE} \
  --preprocessing_num_workers 128 \
  --dataloader_num_workers 30 \
  --dataloader_pin_memory \
  --skip_memory_metrics true \
  --ddp_find_unused_parameters false \
  --output_dir transformers_output_dir \
  --overwrite_output_dir \
  --report_to wandb
