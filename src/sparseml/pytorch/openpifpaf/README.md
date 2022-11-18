# OpenPifPaf Integration

Sample training command:
```bash
python src/sparseml/pytorch/openpifpaf/train.py \
    --lr=0.001 \
    --momentum=0.9 \
    --b-scale=10.0 \
    --clip-grad-value=10.0 \
    --bce-background-clamp=-7 \
    --epochs=100 \
    --lr-decay 80 90 \
    --lr-decay-epochs=10 \
    --batch-size=16 \
    --weight-decay=1e-5 \
    --dataset=cocodet \
    --cocodet-train-image-dir=/network/datasets/coco/images/train2017 \
    --cocodet-val-image-dir=/network/datasets/coco/images/val2017 \
    --cocodet-train-annotations=/network/datasets/coco/annotations/instances_train2017.json \
    --cocodet-val-annotations=/network/datasets/coco/annotations/instances_val2017.json \
    --cocodet-upsample=2 \
    --basenet=mobilenetv3small
```