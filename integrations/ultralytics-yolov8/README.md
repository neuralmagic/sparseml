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

# **SparseML YOLOv8 Integration**

This directory explains how to use SparseML's `ultralytics/ultralytics` integration to train inference-optimized sparse YOLOv8 models on your dataset.

There are two main workflows enabled by SparseML:
- **Sparse Transfer Learning** - fine-tune a pre-sparsified YOLOv8 checkpoint on your own dataset. **[RECOMMENDED]**
- **Sparsification from Scratch** - apply pruning and quantization to sparsify any YOLOv8 model from scratch.

Once trained, SparseML enables you to export models to the ONNX format, such that they can be deployed with DeepSparse.

## **Installation**

Install with `pip`:

```bash
pip install sparseml[ultralytics]
```

## **Tutorials**

- [Sparse Transfer Learning With the CLI](tutorials/sparse-transfer-learning.md) **[HIGHLY RECOMMENDED]**
- Sparsifying From Scratch With the CLI (coming soon!)

## **Quick Tour**

### **SparseZoo**

SparseZoo is an open-source repository of pre-sparsified models, including each version of of YOLOv8. With SparseML, you can fine-tune these pre-sparsified checkpoints onto custom datasets (while maintaining sparsity) via sparse transfer learning. This makes training inference-optimized sparse models almost identical to your typical YOLOv8 training workflow.

[Check out the available models](https://sparsezoo.neuralmagic.com/?useCase=detection&architectures=yolov8)

### **Recipes**

Recipes are YAML files that encode the instructions for sparsifying a model or sparse transfer learning. SparseML accepts the recipes as inputs, parses the instructions, and applies the specified algorithms and hyperparameters during the training process. In this way, recipes are the declarative interface for specifying which sparsity-related algorithms to apply, allowing you to apply sparsity related algorithms from the familiar YOLOv8 training script.

### **SparseML CLI**

SparseML's CLI is built on top of YOLOv8's [`train` cli](https://docs.ultralytics.com/usage/cli/) script. This enables you to kick-off sparse training workflows with all of the friendly utilities from the Ultralytics repository like dataset loading and preprocessing, checkpoint saving, metric reporting, and logging handled for you. Appending the `--help` argument will provide a full list of options for training in SparseML:

```bash
sparseml.ultralytics.train --help
```

output:
```
usage: sparseml.ultralytics.train [-h] [--model MODEL] [--data DATA] [--recipe RECIPE] [--recipe-args RECIPE_ARGS] [--batch BATCH_SIZE] [--epochs EPOCHS] [--imgsz IMGSZ]
                             [--resume] [--patience PAIENCE] [--save] [--cache] [--device [DEVICE]] [--workers WORKERS] [--project [PROJECT]]
                             ...
                            
```

SparseML inherits most arguments from the Ultralytics repository. [Check out the YOLOv8 documentation for usage](https://docs.ultralytics.com/usage/cli/).

## **Quickstart: Sparse Transfer Learning**

### **Sparse Transfer Learning Overview**

Sparse Transfer is very similar to the typical transfer learning process used to train YOLOv8 models, where  a checkpoint pre-trained on COCO is fine-tuned onto a smaller downstream dataset. With Sparse Transfer Learning, however, the fine-tuning process is started from a pre-sparsified checkpoint and maintains sparsity while the training process occurs.

Here, you will fine-tune a [80% pruned version of YOLOv8m](zoo:cv/detection/yolov8-m/pytorch/ultralytics/coco/pruned80-none) trained and sparsified on COCO onto the smaller COCO128 dataset.

### **Kick off Training**

You will use SparseML's `sparseml.ultralytics.train` training script.

To run sparse transfer learning, you first need to create/select a sparsification recipe. For sparse transfer, you need a recipe that instructs SparseML to maintain sparsity during training and to quantize the model over the final epochs. In the SparseZoo, there are several sparse versions of YOLOv8 which were fine-tuned on the VOC dataset. The [80% pruned-quantized version of YOLOv8m](https://sparsezoo.neuralmagic.com/models/yolov8-m-voc_coco-pruned80_quantized) is identified by the following stub:

```bash
zoo:cv/detection/yolov8-m/pytorch/ultralytics/voc/pruned80_quant-none
```

<details>
<summary>Here is what the recipe looks like:</summary>

```yaml
version: 1.1.0

# General variables
num_epochs: 56
init_lr: 1.e-6
final_lr: 1.e-8

# Quantization variables
qat_start_epoch: 50
observer_freeze_epoch: 51
bn_freeze_epoch: 51

training_modifiers:
  - !EpochRangeModifier
    start_epoch: 0
    end_epoch: eval(num_epochs)

  - !LearningRateFunctionModifier
    start_epoch: eval(qat_start_epoch)
    end_epoch: eval(num_epochs)
    lr_func: cosine
    init_lr: eval(init_lr)
    final_lr: eval(final_lr)
 
pruning_modifiers:
  - !ConstantPruningModifier
    start_epoch: 0
    params: ["re:^((?!dfl).)*$"] 

quantization_modifiers:
  - !QuantizationModifier
    start_epoch: eval(qat_start_epoch)
    disable_quantization_observer_epoch: eval(observer_freeze_epoch)
    freeze_bn_stats_epoch: eval(bn_freeze_epoch)
    ignore: ['Upsample', 'Concat', 'model.22.dfl.conv']
    scheme_overrides:
      model.2.cv1.act:
        output_activations:
          num_bits: 8
          symmetric: False
      model.2.m.0.cv1.conv:
        input_activations: null
      model.2.m.0.add_input_0:
        input_activations: null
      model.4.cv1.act:
        output_activations:
          num_bits: 8
          symmetric: False
      model.4.m.0.cv1.conv:
        input_activations: null
      model.4.m.0.add_input_0:
        input_activations: null
      model.4.cv2.act:
        output_activations:
          num_bits: 8
          symmetric: False
      model.5.conv:
        input_activations: null
      model.6.cv1.act:
        output_activations:
          num_bits: 8
          symmetric: False
      model.6.m.0.cv1.conv:
        input_activations: null
      model.6.m.0.add_input_0:
        input_activations: null
      model.6.cv2.act:
        output_activations:
          num_bits: 8
          symmetric: False
      model.7.conv:
        input_activations: null
        output_activations:
          num_bits: 8
          symmetric: False
      model.8.cv1.act:
        output_activations:
          num_bits: 8
          symmetric: False
      model.8.m.0.cv1.conv:
        input_activations: null
      model.8.m.0.add_input_0:
        input_activations: null
      model.8.cv2.act:
        output_activations:
          num_bits: 8
          symmetric: False
      model.9.cv1.act:
        output_activations:
          num_bits: 8
          symmetric: False
      model.9.cv2.act:
        output_activations:
          num_bits: 8
          symmetric: False
      model.12.cv1.act:
        output_activations:
          num_bits: 8
          symmetric: False
      model.12.m.0.cv1.conv:
        input_activations: null
      model.12.m.0.cv2.act:
        output_activations:
          num_bits: 8
          symmetric: False
      model.12.m.1.cv1.conv:
        input_activations: null
      model.12.m.1.cv2.act:
        output_activations:
          num_bits: 8
          symmetric: False
      model.12.cv2.act:
        output_activations:
          num_bits: 8
          symmetric: False
      model.15.cv1.act:
        output_activations:
          num_bits: 8
          symmetric: False
      model.15.m.0.cv1.conv:
        input_activations: null
      model.15.m.0.cv2.act:
        output_activations:
          num_bits: 8
          symmetric: False
      model.15.cv2.act:
        output_activations:
          num_bits: 8
          symmetric: False
      model.15.m.1.cv1.conv:
        input_activations: null
      model.15.m.1.cv2.act:
        output_activations:
          num_bits: 8
          symmetric: False
      model.16.conv:
        input_activations: null
      model.16.act:
        output_activations:
          num_bits: 8
          symmetric: False
      model.18.cv1.act:
        output_activations:
          num_bits: 8
          symmetric: false
      model.18.m.0.cv1.conv:
        input_activations: null
      model.18.m.0.cv2.act:
        output_activations:
          num_bits: 8
          symmetric: False
      model.18.m.1.cv1.conv:
        input_activations: null
      model.18.m.1.cv2.act:
        output_activations:
          num_bits: 8
          symmetric: False
      model.19.act:
        output_activations:
          num_bits: 8
          symmetric: False
      model.21.cv1.act:
        output_activations:
          num_bits: 8
          symmetric: false
      model.21.m.0.cv1.conv:
        input_activations: null
      model.21.m.0.cv2.act:
        output_activations:
          num_bits: 8
          symmetric: False
      model.21.m.1.cv1.conv:
        input_activations: null
      model.21.m.1.cv2.act:
        output_activations:
          num_bits: 8
          symmetric: False
      model.22.cv2.0.0.conv:
        input_activations: null
      model.22.cv3.0.0.conv:
        input_activations: null
```
</details>
   
The key `Modifiers` for sparse transfer learning are the following:
- `ConstantPruningModifier` instructs SparseML to maintain the sparsity structure of the network during the fine-tuning process
- `QuantizationModifier` tells SparseML to quantize the weights with quantization-aware training over the last 5 epochs. The `scheme_overrides` are a bit complicated here to handle the GeLU activations followed by the C2f module introduced in YOLOv8, but you do not need to worry too much about this.

SparseML parses the `Modifers` in the recipe and updates the training loop with logic encoded therein.

Run the following to transfer learn from the 80% pruned YOLOv8m onto the COCO128 dataset:
```bash
sparseml.ultralytics.train \
  --model "zoo:cv/detection/yolov8-m/pytorch/ultralytics/coco/pruned80-none" \
  --recipe "zoo:cv/detection/yolov8-m/pytorch/ultralytics/voc/pruned80_quant-none" \
  --data "coco128.yaml" \
  --batch 2
```

The script uses the SparseZoo stubs to identify and download the starting checkpoint and YAML-based recipe file from the SparseZoo. SparseML parses the transfer learning recipe and adjusts the training process to maintain sparsity during the fine-tuning process. It then kicks off the training process. Note that in the command above, the `--model` has a zoo stub indicating the model was trained on COCO whereas the `--recipe` has a zoo stub indicating the model was trained on VOC. As such, SparseML downloads the 80% pruned model trained on COCO and the sparse transfer learning recipe that was used to fine-tune and quantized the model on VOC.

The resulting model is 80% pruned and quantized and is trained on COCO128!

Of course, you should feel free to download the recipes and edit the hyperparameters as needed!

### **Export to ONNX**

Run the `sparseml.ultralytics.export_onnx` command to export the model to ONNX. Be sure the `--model` argument points to your trained model.

```bash
sparseml.ultralytics.export_onnx \
  --model ./runs/detect/train/weights/last.pt \
  --save_dir yolov8-m
```

### **DeepSparse Deployment**

Once exported to ONNX, you can deploy your models with DeepSparse. Checkout the [DeepSparse Repository](https://github.com/neuralmagic/deepsparse) for more details.

## **Next Steps**

Check out the tutorials for more details on additional functionality like training with other YOLOv8 versions and using custom datasets:

- [Sparse Transfer Learning With the CLI](tutorials/sparse-transfer-learning.md). **[HIGHLY RECOMMENDED]**
