<!--
Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

---
# General Epoch/LR Hyperparams
num_epochs: &num_epochs 50

# Pruning Hyperparams
init_sparsity: &init_sparsity 0.05
pruning_start_epoch: &pruning_start_epoch 5
pruning_end_epoch: &pruning_end_epoch 40
update_frequency: &pruning_update_frequency 1

prune_none_target_sparsity: &prune_none_target_sparsity 0.4
prune_low_target_sparsity: &prune_low_target_sparsity 0.75
prune_mid_target_sparsity: &prune_mid_target_sparsity 0.8
prune_high_target_sparsity: &prune_high_target_sparsity 0.92


# modifiers
training_modifiers:
  - !EpochRangeModifier
    start_epoch: 0.0
    end_epoch: *num_epochs

pruning_modifiers:
  - !ConstantPruningModifier
    start_epoch: 0.0
    params:
      - model.1.conv.weight
      - model.2.cv1.conv.weight
      - model.2.cv2.conv.weight
      - model.3.conv.weight
      - model.4.0.cv1.conv.weight
      - model.4.0.cv2.conv.weight
      - model.4.1.cv1.conv.weight
      - model.4.1.cv2.conv.weight
      - model.5.conv.weight
      - model.6.0.cv1.conv.weight
      - model.6.0.cv2.conv.weight
      - model.6.1.cv1.conv.weight
      - model.6.1.cv2.conv.weight
      - model.6.2.cv1.conv.weight
      - model.6.2.cv2.conv.weight
      - model.6.3.cv1.conv.weight
      - model.6.3.cv2.conv.weight
      - model.6.4.cv1.conv.weight
      - model.6.4.cv2.conv.weight
      - model.6.5.cv1.conv.weight
      - model.6.5.cv2.conv.weight
      - model.6.6.cv1.conv.weight
      - model.6.6.cv2.conv.weight
      - model.6.7.cv1.conv.weight
      - model.6.7.cv2.conv.weight
      - model.7.conv.weight
      - model.8.0.cv1.conv.weight
      - model.8.0.cv2.conv.weight
      - model.8.1.cv1.conv.weight
      - model.8.1.cv2.conv.weight
      - model.8.2.cv1.conv.weight
      - model.8.2.cv2.conv.weight
      - model.8.3.cv1.conv.weight
      - model.8.3.cv2.conv.weight
      - model.8.4.cv1.conv.weight
      - model.8.4.cv2.conv.weight
      - model.8.5.cv1.conv.weight
      - model.8.5.cv2.conv.weight
      - model.8.6.cv1.conv.weight
      - model.8.6.cv2.conv.weight
      - model.8.7.cv1.conv.weight
      - model.8.7.cv2.conv.weight
      - model.9.conv.weight
      - model.10.0.cv1.conv.weight
      - model.10.0.cv2.conv.weight
      - model.10.1.cv1.conv.weight
      - model.10.1.cv2.conv.weight
      - model.10.2.cv1.conv.weight
      - model.10.2.cv2.conv.weight
      - model.10.3.cv1.conv.weight
      - model.10.3.cv2.conv.weight
      - model.11.cv1.conv.weight
      - model.11.cv2.conv.weight
      - model.12.cv1.conv.weight
      - model.12.cv2.conv.weight
      - model.13.conv.weight
      - model.14.conv.weight
      - model.15.conv.weight
      - model.16.conv.weight
      - model.19.cv1.conv.weight
      - model.19.cv2.conv.weight
      - model.20.cv1.conv.weight
      - model.20.cv2.conv.weight
      - model.21.conv.weight
      - model.22.conv.weight
      - model.23.conv.weight
      - model.26.cv1.conv.weight
      - model.26.cv2.conv.weight
      - model.27.0.cv1.conv.weight
      - model.27.0.cv2.conv.weight
      - model.27.1.cv1.conv.weight
      - model.27.1.cv2.conv.weight

  - !GMPruningModifier
    params:
      - model.28.m.0.weight
      - model.28.m.1.weight
      - model.28.m.2.weight
    init_sparsity: *init_sparsity
    final_sparsity: *prune_low_target_sparsity
    start_epoch: *pruning_start_epoch
    end_epoch: *pruning_end_epoch
    update_frequency: *pruning_update_frequency
---

# YOLOv3-SPP Pruned Transfer Learning

This recipe transfer learns from a sparse, [YOLOv3-SPP](https://arxiv.org/abs/1804.02767) model.
It was originally tested on the VOC dataset and achieved 0.84 mAP@0.5.

Training was done using 4 GPUs at half precision with the [SparseML integration with ultralytics/yolov3](https://github.com/neuralmagic/sparseml/tree/main/integrations/ultralytics-yolov3).

When running, adjust hyperparameters based on training environment and dataset.

## Weights and Biases

The training results for this recipe are made available through Weights and Biases for easy viewing.

- [YOLOv3-SPP LeakyReLU VOC Transfer Learning](https://wandb.ai/neuralmagic/yolov3-spp-voc-sparse-transfer-learning/runs/3kvb4neh)

## Training

To set up the training environment, follow the instructions on the [integration README](https://github.com/neuralmagic/sparseml/blob/main/integrations/ultralytics-yolov3/README.md).
Using the given training script from the `yolov3` directory the following command can be used to launch this recipe.  
Adjust the script command for your GPU device setup. 
Ultralytics supports both DataParallel and DDP.

The sparse weights used with this recipe are stored in the SparseZoo and can be retrieved with the following code:
 ```python
from sparsezoo import Model

stub = 'zoo:cv/detection/yolo_v3-spp/pytorch/ultralytics/coco/pruned-aggressive_97'
model = Model(stub)
downloaded_path = model.path
print(f'Model with stub {stub} downloaded to {downloaded_path}.')
```

*script command:*

```bash
python train.py \
    --data voc.yaml \
    --cfg ../models/yolov3-spp.lrelu.yaml \
    --weights DOWNLOADED_PATH \
    --hyp data/hyp.finetune.yaml \
    --recipe ../recipes/yolov3-spp.transfer_learn_pruned.md
```
