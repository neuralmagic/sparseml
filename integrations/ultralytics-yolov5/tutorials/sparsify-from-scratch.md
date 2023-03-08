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

# Sparsifying YOLOv5 From Scratch

This page explains how to sparsify a YOLOv5 model from scratch with SparseML's CLI.

## Overview

Sparsifying a model involves removing redundant information from a 
trained model using algorithms such as pruning and quantization. The sparse models can then be deployed with DeepSparse, which implements many optimizations to take advantage of sparsity to gain a performance speedup.

In this tutorial, we will demonstrate how to use recipes to create 
sparse versions of YOLOv5.

**Pro Tip**: For YOLOv5, there are pre-sparsified checkpoints of each version available in [SparseZoo](https://sparsezoo.neuralmagic.com/?domain=cv&sub_domain=detection&page=1). 
***We highly recommend using the [Sparse Transfer Learning](sparse-transfer-learning.md) pathway to fine-tune one of these checkpoints onto your dataset 
rather than sparsifying from scratch.***

## Sparsification Recipes

Recipes are SparseML's YAML-based declarative interface for specifying the sparsity-related algorithms and parameters that should be applied during the training 
process. The SparseML CLI script parses the recipes and modifies the YOLOv5 training loop to implement the specified algorithms.

The SparseZoo hosts the recipes that were used to create the sparse versions of YOLOv5.

<details>
   <summary> Click to see the recipe used to create the 75% pruned-quantized YOLOv5s</summary>

```yaml
version: 1.1.0

# General Hyperparams
num_epochs: 250
init_lr: 0.01
final_lr: 0.002
weights_warmup_lr: 0
biases_warmup_lr: 0.1

# Pruning Hyperparams
init_sparsity: 0.05
pruning_start_epoch: 4
pruning_end_epoch: 150
pruning_update_frequency: 1.0


# Quantization variables
quantization_epochs: 10
quantization_lr: 1.e-3

# Knowledge distillation variables
per_layer_distillation_gain: 0.01

#Modifiers
training_modifiers:
  - !EpochRangeModifier
    start_epoch: 0
    end_epoch: eval(num_epochs)

  - !LearningRateFunctionModifier
    start_epoch: 3
    end_epoch: eval(num_epochs - quantization_epochs)
    lr_func: linear
    init_lr: eval(init_lr)
    final_lr: eval(final_lr)

  - !LearningRateFunctionModifier
    start_epoch: 0
    end_epoch: 3
    lr_func: linear
    init_lr: eval(weights_warmup_lr)
    final_lr: eval(init_lr)
    param_groups: [0, 1]

  - !LearningRateFunctionModifier
    start_epoch: 0
    end_epoch: 3
    lr_func: linear
    init_lr: eval(biases_warmup_lr)
    final_lr: eval(init_lr)
    param_groups: [2]

  - !LearningRateFunctionModifier
    start_epoch: eval(num_epochs - quantization_epochs)
    end_epoch: eval(num_epochs)
    lr_func: cosine
    init_lr: eval(quantization_lr)
    final_lr: 1.e-9
 
pruning_modifiers:
  - !GMPruningModifier
    params:
      - model.13.cv1.conv.weight
      - model.13.cv2.conv.weight
      - model.13.cv3.conv.weight
      - model.13.m.0.cv1.conv.weight
      - model.14.conv.weight
      - model.17.cv1.conv.weight
      - model.17.cv2.conv.weight
      - model.17.cv3.conv.weight
      - model.17.m.0.cv1.conv.weight
      - model.2.cv1.conv.weight
      - model.2.cv2.conv.weight
      - model.2.cv3.conv.weight
      - model.2.m.0.cv1.conv.weight
      - model.20.cv1.conv.weight
      - model.20.cv2.conv.weight
      - model.20.cv3.conv.weight
      - model.23.cv1.conv.weight
      - model.23.cv2.conv.weight
      - model.23.m.0.cv1.conv.weight
      - model.24.m.2.weight
      - model.4.cv1.conv.weight
      - model.4.cv2.conv.weight
      - model.4.cv3.conv.weight
      - model.4.m.0.cv1.conv.weight
      - model.4.m.1.cv1.conv.weight
      - model.4.m.1.cv2.conv.weight
      - model.6.cv1.conv.weight
      - model.6.cv2.conv.weight
      - model.6.cv3.conv.weight
      - model.8.cv1.conv.weight
      - model.8.cv2.conv.weight
      - model.8.cv3.conv.weight
      - model.8.m.0.cv1.conv.weight
      - model.9.cv1.conv.weight    
    init_sparsity: eval(init_sparsity)
    final_sparsity: 0.4240  
    start_epoch: eval(pruning_start_epoch)
    end_epoch: eval(pruning_end_epoch)
    update_frequency: eval(pruning_update_frequency)
        
  - !GMPruningModifier
    params:
      - model.20.m.0.cv1.conv.weight
      - model.4.m.0.cv2.conv.weight
      - model.9.cv2.conv.weight    
    init_sparsity: eval(init_sparsity)
    final_sparsity: 0.4838  
    start_epoch: eval(pruning_start_epoch)
    end_epoch: eval(pruning_end_epoch)
    update_frequency: eval(pruning_update_frequency)
        
  - !GMPruningModifier
    params:
      - model.18.conv.weight
      - model.2.m.0.cv2.conv.weight
      - model.6.m.2.cv2.conv.weight
    init_sparsity: eval(init_sparsity)
    final_sparsity: 0.5374  
    start_epoch: eval(pruning_start_epoch)
    end_epoch: eval(pruning_end_epoch)
    update_frequency: eval(pruning_update_frequency)

  - !GMPruningModifier
    params:
      - model.0.conv.weight
      - model.24.m.0.weight
      - model.3.conv.weight
    init_sparsity: eval(init_sparsity)
    final_sparsity: 0.5854  
    start_epoch: eval(pruning_start_epoch)
    end_epoch: eval(pruning_end_epoch)
    update_frequency: eval(pruning_update_frequency)

  - !GMPruningModifier
    params:
      - model.13.m.0.cv2.conv.weight
      - model.24.m.1.weight
      - model.5.conv.weight
      - model.6.m.1.cv2.conv.weight
    init_sparsity: eval(init_sparsity)
    final_sparsity: 0.6284  
    start_epoch: eval(pruning_start_epoch)
    end_epoch: eval(pruning_end_epoch)
    update_frequency: eval(pruning_update_frequency) 

  - !GMPruningModifier
    params:
      - model.1.conv.weight
      - model.17.m.0.cv2.conv.weight
      - model.20.m.0.cv2.conv.weight
    init_sparsity: eval(init_sparsity)
    final_sparsity: 0.7325  
    start_epoch: eval(pruning_start_epoch)
    end_epoch: eval(pruning_end_epoch)
    update_frequency: eval(pruning_update_frequency)
    
  - !GMPruningModifier
    params:
      - model.23.cv3.conv.weight
      - model.6.m.0.cv2.conv.weight
      - model.7.conv.weight
      - model.8.m.0.cv2.conv.weight 
    init_sparsity: eval(init_sparsity)
    final_sparsity: 0.7602  
    start_epoch: eval(pruning_start_epoch)
    end_epoch: eval(pruning_end_epoch)
    update_frequency: eval(pruning_update_frequency)
   
  - !GMPruningModifier
    params:
      - model.23.m.0.cv2.conv.weight
    init_sparsity: eval(init_sparsity)
    final_sparsity: 0.8453
    start_epoch: eval(pruning_start_epoch)
    end_epoch: eval(pruning_end_epoch)
    update_frequency: eval(pruning_update_frequency) 

  - !GMPruningModifier
    params:
      - model.21.conv.weight 
    init_sparsity: eval(init_sparsity)
    final_sparsity: 0.9002  
    start_epoch: eval(pruning_start_epoch)
    end_epoch: eval(pruning_end_epoch)
    update_frequency: eval(pruning_update_frequency)

knowledge_disitillation_modifiers:
  - !PerLayerDistillationModifier
    start_epoch: 0.0
    end_epoch: eval(num_epochs - quantization_epochs)
    gain: eval(per_layer_distillation_gain)
    project_features: true
    student_layer_names:
    - model.0
    - model.1
    - model.2.cv1
    - model.2.cv2
    - model.2.cv3
    - model.2.m.0.cv1
    - model.2.m.0.cv2
    - model.3
    - model.4.cv1
    - model.4.cv2
    - model.4.cv3
    - model.4.m.0.cv1
    - model.4.m.0.cv2
    - model.5
    - model.6.cv1
    - model.6.cv2
    - model.6.cv3
    - model.6.m.0.cv1
    - model.6.m.0.cv2
    - model.7
    - model.8.cv1
    - model.8.cv2
    - model.8.cv3
    - model.8.m.0.cv1
    - model.8.m.0.cv2
    - model.9.cv1
    - model.9.cv2
    - model.10
    - model.13.cv1
    - model.13.cv2
    - model.13.cv3
    - model.13.m.0.cv1
    - model.13.m.0.cv2
    - model.14
    - model.17.cv1
    - model.17.cv2
    - model.17.cv3
    - model.17.m.0.cv1
    - model.17.m.0.cv2
    - model.18
    - model.20.cv1
    - model.20.cv2
    - model.20.cv3
    - model.20.m.0.cv1
    - model.20.m.0.cv2
    - model.21
    - model.23.cv1
    - model.23.cv2
    - model.23.cv3
    - model.23.m.0.cv1
    - model.23.m.0.cv2
    - model.24.m.0
    - model.24.m.1
    - model.24.m.2

quantization_modifiers:
  - !QuantizationModifier
    start_epoch: eval(num_epochs - quantization_epochs)
    submodules:
      - model
    custom_quantizable_module_types: ['SiLU']
    exclude_module_types: ['SiLU']
    quantize_conv_activations: False
    disable_quantization_observer_epoch: eval(num_epochs - quantization_epochs + 2)
    freeze_bn_stats_epoch: eval(num_epochs - quantization_epochs + 1)
```
   
There is is a lot here, but the important items are the `pruning_modifiers`, `distillation_modifiers`, and
`quantization_modifiers`.

The `pruning_modifiers` instruct SparseML to apply the Global Magnitude Pruning algorithm to various layers of
the network. As you can see, the recipe specifies a target level of sparsity for each layer of the network.
At the end of every epoch, the GMP algorithm iteratively removes the lowest magnitude weights gradually inducing
sparsitty into the network.

The `distillation_modifiers` instruct SparseML to apply model distillation from a teacher during the 
pruning process. In this case, we perform per-layer distillation. The teacher model helps to improve
accuracy during the pruning process.

The `quantization_modifiers` instruct SparseML to apply Quantization Aware Training during the final
few epochs, creating a sparse version of the model.

</details>

## Applying a Recipe

We can use the SparseML CLI to apply this recipe to a YOLOv5 model.

Run the following:

```bash
sparseml.yolov5.train \
  --weights zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/base-none \
  --recipe zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned75_quant-none \
  --teacher-weights zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/base-none \
  --data coco.yaml \
  --hyp hyps/hyp.scratch-low.yaml --cfg yolov5s.yaml --patience 0 --gradient-accum-steps 4
```

Let's discuss the key arguments:

- `--weights zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/base-none` specifies the starting checkpoint for the sparsification. Here, we passed a SparseZoo stub identifying the standard dense YOLOv5s model in SparseZoo. Alternatively, you can pass a path to a YOLOv5 PyTorch model.

- `--recipe zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned75_quant-none` specifies the sparsification recipe to be applied. Here, we passed a SparseZoo
stub identifying the sparsification recipe shown above in the SparseZoo. Alternatively, you can pass a path to a local YAML-based recipe.

- `--teacher-weights zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/base-none` specifies the teacher model to be used for model distillation. This is an
optional argument (as pruning can be applied without model distillation as well). Here, we passed a SparseZoo stub identifying the standard dense YOLOv5s model in SparseZoo. Alternatively, you can pass a path to a YOLOv5 PyTorch model.

- `--data coco.yaml` specifies dataset configuration file to use during the sparsification process. Here, we pass `coco.yaml`, which SparseML instructs SparseML to 
automatically download the COCO dataset. Alternatively, you can pass a config file for your local dataset. Checkout the [Ultralytics Custom Data Tutorial Repo](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data) for more details on how to structure your datasets. SparseML conforms to the Ultralytics specification.

- `--hyp hyps/hyp.scratch-low.yaml` specifies a path to the hyperparameters for the training. Here, we use a [built in configuration](https://github.com/neuralmagic/yolov5/blob/master/data/hyps/hyp.scratch-low.yaml). Note that any hyperparameters specified in the `--recipe` (e.g. epochs or learning rate) will override anything passed to the `--hyps` argument. For instance, in this case, the recipe specifies the learning rate schedule. The specification in the recipe overrides the lr in the hyperparameter file.

At the end, you will have a 75% pruned and quantized version of YOLOv5s trained on COCO! This model achieves 54.69 mAP@0.5.

## Exporting for Inference

Once trained, you can export the model to ONNX for inference with DeepSparse. 

Run the following:

```bash 
sparseml.yolov5.export_onnx \
  --weights yolov5_runs/train/exp/weights/last.pt \
  --dynamic
```

The resulting ONNX file is saved in your local directory.

## Other YOLOv5 Models

Here are some sample transfer learning commands for other versions of YOLOv5. Checkout the [SparseZoo](https://sparsezoo.neuralmagic.com/?page=1&domain=cv&sub_domain=detection) for the full repository of pre-sparsified checkpoints.

   - YOLOv5n: Recipe Coming Soon
   
   - YOLOv5s: 75% Pruned-Quantized
```bash
sparseml.yolov5.train \
  --cfg yolov5s.yaml \
  --teacher-weights zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/base-none \
  --data coco.yaml \
  --recipe zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned75_quant-none \
  --hyp hyps/hyp.scratch-low.yaml \
  --weights zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/base-none \
  --patience 0 \
  --gradient-accum-steps 4
```

   - YOLOv5m: Recipe Coming Soon
   
   - YOLOv5l: 90% Pruned-Quantized

```bash
sparseml.yolov5.train \
  --cfg yolov5l.yaml \
  --teacher-weights zoo:cv/detection/yolov5-l/pytorch/ultralytics/coco/base-none \
  --data coco.yaml \
  --recipe zoo:cv/detection/yolov5-l/pytorch/ultralytics/coco/pruned90_quant-none \
  --hyp hyps/hyp.scratch-high.yaml \
  --weights zoo:cv/detection/yolov5-l/pytorch/ultralytics/coco/base-none \
  --patience 0 \
  --gradient-accum-steps 4
```
   - YOLOv5x: Recipe Coming Soon

## Next Steps

Checkout the DeepSparse repository for more information on deploying your sparse YOLOv5 models with DeepSparse for GPU class performance on CPUs.
