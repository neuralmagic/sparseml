# OpenPifPaf Integration

Sample training command:
```bash
CUDA_VISIBLE_DEVICES=0 python src/sparseml/pytorch/openpifpaf/train.py \
    --recipe prune-openpifpaf.yaml
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

Where `prune-openpifpaf.yaml` contains:
```yaml
num_epochs: &num_epochs 10

training_modifiers:
  - !EpochRangeModifier
    start_epoch: 0.0
    end_epoch: *num_epochs

pruning_modifiers:
  - !GMPruningModifier
    params: __ALL_PRUNABLE__
    init_sparsity: 0.05
    final_sparsity: 0.8
    start_epoch: 0.1
    end_epoch: 8.0
    update_frequency: 0.5
```