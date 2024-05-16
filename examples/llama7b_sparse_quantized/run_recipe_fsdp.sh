#!/bin/bash
FSDP_CONFIG='fsdp_config.yaml'
TRAIN_SCRIPT='llama7b_sparse_w4a16.py'

accelerate launch --config_file $FSDP_CONFIG $TRAIN_SCRIPT