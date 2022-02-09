export CUDA_VISIBLE_DEVICES="4,5"

python integrations/pytorch/train.py \
    --recipe-path ac_dc_recipy.md \
    --pretrained False \
    --arch-key /home/dan/ILSVRC \
    --dataset imagenet \
    --dataset-path data \
    --train-batch-size 256 \
    --test-batch-size 256 \
    --loader-num-workers 0 \
    --init-lr 0.256 \
    --optim SGD \
    --optim-args '{"momentum": 0.875, "weight_decay": 0.00003051757813}'

    # /home/dan/ILSVRC