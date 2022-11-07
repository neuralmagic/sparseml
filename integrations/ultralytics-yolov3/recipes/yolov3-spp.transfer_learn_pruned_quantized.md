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
num_epochs: &num_epochs 52
warmup_lr: &warmup_lr 0.0
warmup_lr_bias: &warmup_lr_bias 0.05
init_lr: &init_lr 0.0032
final_lr: &final_lr 0.00038

# Pruning Hyperparams
init_sparsity: &init_sparsity 0.05
pruning_start_epoch: &pruning_start_epoch 5
pruning_end_epoch: &pruning_end_epoch 40
update_frequency: &pruning_update_frequency 1
mask_type: &mask_type [1, 4]

prune_none_target_sparsity: &prune_none_target_sparsity 0.4
prune_low_target_sparsity: &prune_low_target_sparsity 0.75
prune_mid_target_sparsity: &prune_mid_target_sparsity 0.8
prune_high_target_sparsity: &prune_high_target_sparsity 0.92

# Quantization Hyperparams
quantization_start_epoch: &quantization_start_epoch 50
quantization_init_lr: &quantization_init_lr 0.0038

# modifiers
training_modifiers:
  - !EpochRangeModifier
    start_epoch: 0.0
    end_epoch: *num_epochs
    
  - !LearningRateFunctionModifier
    start_epoch: 100
    end_epoch: 160
    lr_func: cosine
    init_lr: *init_lr
    final_lr: *final_lr
    
  - !LearningRateFunctionModifier
    start_epoch: 0
    end_epoch: 3
    lr_func: linear
    init_lr: *warmup_lr
    final_lr: *init_lr
    param_groups: [0, 1]
    
  - !LearningRateFunctionModifier
    start_epoch: 0
    end_epoch: 3
    lr_func: linear
    init_lr: *warmup_lr_bias
    final_lr: *init_lr
    param_groups: [2]
    
  - !SetLearningRateModifier
    start_epoch: *quantization_start_epoch
    learning_rate: *quantization_init_lr

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
    mask_type: *mask_type
        
quantization_modifiers:
  - !QuantizationModifier
    start_epoch: *quantization_start_epoch
    submodules:
      - model.0
      - model.1 
      - model.2
      - model.3 
      - model.4
      - model.5
      - model.6
      - model.7
      - model.8
      - model.9
      - model.10
      - model.11
      - model.12
      - model.13
      - model.14
      - model.15
      - model.16
      - model.17
      - model.18
      - model.19
      - model.20
      - model.21
      - model.22
      - model.23
      - model.24
      - model.25
      - model.26
      - model.27
---

# YOLOv3-SPP Pruned-Quantized Transfer Learning

This recipe transfer learns from a sparse-quantized, [YOLOv3-SPP](https://arxiv.org/abs/1804.02767) model.
It was originally tested on the VOC dataset and achieved 0.838 mAP@0.5.

Training was done using 4 GPUs at half precision with the [SparseML integration with ultralytics/yolov3](https://github.com/neuralmagic/sparseml/tree/main/integrations/ultralytics-yolov3).

When running, adjust hyperparameters based on training environment and dataset.

## Weights and Biases

The training results for this recipe are made available through Weights and Biases for easy viewing.

- [YOLOv3-SPP LeakyReLU VOC Transfer Learning](https://wandb.ai/neuralmagic/yolov3-spp-voc-sparse-transfer-learning/runs/2hbvu7w2)

## Training

To set up the training environment, follow the instructions on the [integration README](https://github.com/neuralmagic/sparseml/blob/main/integrations/ultralytics-yolov3/README.md).
Using the given training script from the `yolov3` directory the following command can be used to launch this recipe.  
Adjust the script command for your GPU device setup. 
Ultralytics supports both DataParallel and DDP.

The sparse weights used with this recipe are stored in the SparseZoo and can be retrieved with the following code:
 ```python
from sparsezoo import Model

stub = 'zoo:cv/detection/yolo_v3-spp/pytorch/ultralytics/coco/pruned_quant-aggressive_94'
model = Model(stub)
downloded_path = model.path
print(f"Model with stub {stub} downloaded to {downloaded_path}.")
```

*script command:*

```bash
python train.py \
    --data voc.yaml \
    --cfg ../models/yolov3-spp.lrelu.yaml \
    --weights DOWNLOADED_PATH \
    --hyp data/hyp.finetune.yaml \
    --recipe ../recipes/yolov3-spp.transfer_learn_pruned_quantized.md
```
