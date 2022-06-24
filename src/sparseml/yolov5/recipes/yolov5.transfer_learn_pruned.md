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
init_lr: &init_lr 0.0032
final_lr: &final_lr 0.000384
warmup_epochs: &warmup_epochs 2
weights_warmup_lr: &weights_warmup_lr 0
biases_warmup_lr: &biases_warmup_lr 0.05

# modifiers
training_modifiers:
  - !EpochRangeModifier
    start_epoch: 0.0
    end_epoch: *num_epochs
    
  - !LearningRateFunctionModifier
    start_epoch: *warmup_epochs
    end_epoch: *num_epochs
    lr_func: cosine
    init_lr: *init_lr
    final_lr: *final_lr
    
  - !LearningRateFunctionModifier
    start_epoch: 0
    end_epoch: *warmup_epochs
    lr_func: linear
    init_lr: *weights_warmup_lr
    final_lr: *init_lr
    param_groups: [0, 1]
    
  - !LearningRateFunctionModifier
    start_epoch: 0
    end_epoch: *warmup_epochs
    lr_func: linear
    init_lr: *biases_warmup_lr
    final_lr: *init_lr
    param_groups: [2]

pruning_modifiers:
  - !ConstantPruningModifier
    start_epoch: 0.0
    params: __ALL_PRUNABLE__
---

# YOLOv5 Pruned Transfer Learning

This recipe transfer learns from a sparse, [YOLOv5](https://github.com/ultralytics/yolov5) model.
It has been tested with s and l versions on the VOC dataset using the the [SparseML integration with ultralytics/yolov5](https://github.com/neuralmagic/sparseml/tree/main/integrations/ultralytics-yolov5).

When running, adjust hyperparameters based on training environment and dataset.

## Weights and Biases

The training results for this recipe are made available through Weights and Biases for easy viewing.

- [YOLOv5 VOC Transfer Learning](https://wandb.ai/neuralmagic/yolov5-voc-sparse-transfer-learning)

## Training

To set up the training environment, follow the instructions on the [integration README](https://github.com/neuralmagic/sparseml/blob/main/integrations/ultralytics-yolov5/README.md).
Using the given training script from the `yolov5` directory the following command can be used to launch this recipe.  
Adjust the script command for your GPU device setup. 
Ultralytics supports both DataParallel and DDP.
Finally, the sparse weights used with this recipe are stored in the SparseZoo and can be retrieved by passing in a SparseZoo stub to the `--weights` argument.

*script command:*

```bash
python train.py \
    --data voc.yaml \
    --cfg ../models/yolov5s.yaml \
    --weights zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned-aggressive_96 \
    --hyp data/hyp.finetune.yaml \
    --recipe ../recipes/yolov5s.transfer_learn_pruned.md
```