#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

ROOT=$HOME/src/neuralmagic/sparseml/src/sparseml/experimental/sparsegpt

DATASET=c4

RECIPE_DIR=$ROOT/recipes
RECIPE_NAME=opt-1.3b-opt_pretrain-pruned50_quantW8A8

SRC_MODEL_ORG=facebook
SRC_MODEL_NAME=opt-1.3b
SRC_MODEL=$SRC_MODEL_ORG/$SRC_MODEL_NAME

SP=0.5
WBITS=8

ID=$RANDOM

SMOOTH=0
SMOOTH_DIR=$HOME/src/smoothquant/act_scales
SMOOTH_FILE=$SMOOTH_DIR/$SRC_MODEL_NAME.pt

PTQ=1

TRUE_SEQ=0

DST_MODEL_DIR=$HOME/models/opt
DST_MODEL_NAME=sparsegpt@$SRC_MODEL_NAME@$DATASET@$RECIPE_NAME@SP$SP@SQ$SMOOTH@SEQ$TRUE_SEQ@PTQ$PTQ@ID$ID
DST_MODEL=$DST_MODEL_DIR/$DST_MODEL_NAME

EVAL_DENSE=0

OBSERVER_BATCHES=100

python $ROOT/main.py $SRC_MODEL $DATASET \
       --data-sequence-length 2048 \
       --sequential_hessian_within_layer $TRUE_SEQ \
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
