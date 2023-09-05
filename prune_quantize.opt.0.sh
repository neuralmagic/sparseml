#!/bin/bash

# venv: /home/sadkins/sparseml/.venv

export CUDA_VISIBLE_DEVICES=0

ROOT=/home/sadkins/sparseml/src/sparseml/experimental/sparsegpt

source ./scripts/start_here.sh

DATASET=c4

RECIPE_DIR=./recipes
RECIPE_NAME=opt350m.W8A8linear.A8A8O16matmul

SRC_MODEL_ORG=facebook
SRC_MODEL_NAME=opt-350m
SRC_MODEL=$SRC_MODEL_ORG/$SRC_MODEL_NAME

SP=0.5
WBITS=8

ID=$RANDOM

SMOOTH=0
SMOOTH_DIR=$ROOT/src/natuan/smoothquant/act_scales
SMOOTH_FILE=$SMOOTH_DIR/$SRC_MODEL_NAME.pt

PTQ=0

DST_MODEL_DIR=/home/sadkins/sparseml/export/opt
DST_MODEL_NAME=sparsegpt@$SRC_MODEL_NAME@$DATASET@$RECIPE_NAME@SP$SP@SQ$SMOOTH@PTQ$PTQ@ID$ID
DST_MODEL=$DST_MODEL_DIR/$DST_MODEL_NAME

EVAL_DENSE=0

OBSERVER_BATCHES=100

/home/sadkins/sparseml/.venv/bin/python3 $ROOT/main.py $SRC_MODEL $DATASET \
       --data-sequence-length 2048 \
       --recipe $RECIPE_DIR/$RECIPE_NAME.md \
       --sparsity $SP \
       --eval-dense $EVAL_DENSE \
       --wbits $WBITS \
       --observer-batches $OBSERVER_BATCHES \
       --ptq $PTQ \
       --ptq-init 1 \
       --smoothquant $SMOOTH \
       --smooth-activation-file $SMOOTH_FILE \
       --save $DST_MODEL

cp "$0" $DST_MODEL/command.sh
