# OpenPifPaf Integration

## Training

Sample training command:
```bash
CUDA_VISIBLE_DEVICES=0 python src/sparseml/openpifpaf/train.py \
    --recipe prune-openpifpaf.yaml \
    --lr=0.001 \
    --momentum=0.9 \
    --b-scale=10.0 \
    --clip-grad-value=10.0 \
    --bce-background-clamp=-7 \
    --epochs=100 \
    --lr-decay 230 240 \
    --lr-decay-epochs=10 \
    --batch-size=8 \
    --weight-decay=1e-5 \
    --dataset=cocokp \
    --cocokp-square-edge=513 \
    --cocokp-upsample=2 \
    --cocokp-extended-scale \
    --cocokp-orientation-invariant=0.1 \
    --cocokp-train-image-dir=/network/datasets/coco/images/train2017 \
    --cocokp-val-image-dir=/network/datasets/coco/images/val2017 \
    --cocokp-train-annotations=/home/corey/coco/annotations/person_keypoints_train2017.json \
    --cocokp-val-annotations=/home/corey/coco/annotations/person_keypoints_val2017.json \
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

## Exporting

From a training checkpoint
```bash
python src/sparseml/openpifpaf/export.py \
    --checkpoint=<path to .pkl.epochXXX file from train> \
    --dataset=cocokp
```
