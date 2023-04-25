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
# Epoch and Learning-Rate variables
num_epochs: &num_epochs 10.0
init_lr: &init_lr 0.008
lr_step_milestones: &lr_step_milestones [5]

# quantization variables
quantization_start_epoch: &quantization_start_epoch 4.0

training_modifiers:
  - !EpochRangeModifier
    start_epoch: 0.0
    end_epoch: *num_epochs
    
  - !LearningRateModifier
    start_epoch: 0.0
    lr_class: MultiStepLR
    lr_kwargs:
      milestones: *lr_step_milestones
      gamma: 0.1
    init_lr: *init_lr

# Phase 1 Sparse Transfer Learning / Recovery
sparse_transfer_learning_modifiers:
  - !ConstantPruningModifier
    start_epoch: 0.0
    params: __ALL_PRUNABLE__

# phase 2 apply quantization
sparse_quantized_transfer_learning_modifiers:
  - !QuantizationModifier
    start_epoch: *quantization_start_epoch

---
# Transfer Learning

This recipe provides a framework for performing sparse transfer learning on
SparseML's `sparsified` ResNet-50, or MobileNet model.  This recipe was written for transfer learning
onto the [Imagenette](https://github.com/fastai/imagenette) dataset using
a single GPU, Adam optimizer, and batch size 32.  Adjust the hyperparameters at the top of the file accordingly
for your dataset and training environment.

## Imagenette
[Imagenette](https://github.com/fastai/imagenette) is a subset of 10 easily classified classes from ImageNet (tench, English springer, cassette player, chain saw, church, French horn, garbage truck, gas pump, golf ball, parachute).


To transfer learn ResNet-50 on [Imagenette](https://github.com/fastai/imagenette) with this recipe, run the following example command.

```
sparseml.image_classification.train \
    --recipe-path integrations/pytorch/recipes/classification.transfer_learn_pruned_quantized.md \
    --checkpoint-path /PATH/TO/MODEL_CHECKPOINT \
    --arch-key resnet50 \
    --model-kwargs '{"ignore_error_tensors": ["classifier.fc.weight", "classifier.fc.bias"]}' \
    --dataset imagenette \
    --dataset-path /PATH/TO/IMAGENETTE  \
    --train-batch-size 32 --test-batch-size 64 \
    --loader-num-workers 0 \
    --optim Adam \
    --optim-args '{}' \
    --model-tag resnet50-imagenette-sparse-transfer-learned-quantized
```

To transfer learn MobileNet on [Imagenette](https://github.com/fastai/imagenette) with this recipe, run the following example command.

```
sparseml.image_classification.train \
    --recipe-path  integrations/pytorch/recipes/classification.transfer_learn_pruned_quantized.md \
    --checkpoint-path /PATH/TO/MODEL_CHECKPOINT \
    --arch-key mobilenet \
    --model-kwargs '{"ignore_error_tensors": ["classifier.fc.weight", "classifier.fc.bias"]}' \
    --dataset imagenette \
    --dataset-path /PATH/TO/IMAGENETTE  \
    --train-batch-size 32 --test-batch-size 64 \
    --loader-num-workers 0 \
    --optim Adam \
    --optim-args '{}' \
    --model-tag mobilenet-imagenette-sparse-transfer-learned-quantized
```
## Other Datasets
To transfer learn this sparsified model to other datasets
you may have to adjust certain hyperparameters in this recipe and/or training script.
Some considerations:
* For more complex datasets, increase the number of epochs, adjusting the learning rate step accordingly
* Adding more learning rate step milestones can lead to more jumps in accuracy
* Increase the learning rate when increasing batch size
* Increasing the number of epochs before starting quantization can give a higher baseline
* If accuracy is not recovering in the quantization phase, the total number of epochs should be increased as well
* Increase the number of epochs if using SGD instead of the Adam optimizer
* Update the base learning rate based on the number of steps needed to train your dataset 
