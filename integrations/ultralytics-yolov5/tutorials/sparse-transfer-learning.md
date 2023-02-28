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

# Sparse Transfer Learning With YOLOv5

This page explains how to fine-tune a pre-sparsified YOLOv5 model with SparseML's CLI.

## Overview

Sparse Transfer is quite similiar to the typical YOLOv5 training, where we fine-tune a checkpoint pretrained on COCO onto a smaller downstream dataset. 

With Sparse Transfer Learning, we simply start the fine-tuning process from a pre-sparsified YOLOv5 and maintain sparsity while the training process occurs.
[SparseZoo](https://sparsezoo.neuralmagic.com/?domain=cv&sub_domain=detection&page=1) contains pre-sparsified checkpoints for each version of YOLOv5, granting
you maximum flexibility in training inference-optimized sparse models.

Let's walk through a couple examples.

## Installations

Install via `pip`:
```pip install sparseml[torchvision]```

## Sparse Transfer Learning with VOC

Let's walk through some a quick example of Sparse Transfer Learning YOLOv5s onto the VOC dataset.

Run the following:
```
sparseml.yolov5.train \
  --weights zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned75_quant-none?recipe_type=transfer_learn \
  --recipe zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned75_quant-none?recipe_type=transfer_learn \
  --data VOC.yaml \
  --patience 0 \
  --cfg yolov5s.yaml \
  --hyp hyps/hyp.finetune.yaml
```

Lets disucss the key arguments:
- `--weights` specifies the starting checkpoint for the training process. Here, we passed a SparseZoo stub, which
identifies the 75% pruned-quantized YOLOv5s model in the SparseZoo. The script downloads the PyTorch model to begin training.

- `--recipe` specifies the transfer learning recipe. Recipes are YAML files that declare the sparsity related algorithms
 that SparseML should apply. For transfer learning, the recipe instructs SparseML to maintian sparsity during training
 and to apply quantization over the final epochs. In this case, we passed a SparseZoo stub, which instructs SparseML
 to download a premade YOLOv5s transfer learning recipe.

- `--data` specifies the dataset. Here, the script automatically downloads the VOC dataset.

Here's what the recipe looks like:

```yaml
version: 1.1.0

# General variables
num_epochs: 55
quantization_epochs: 5
quantization_lr: 1.e-5
final_lr: 1.e-9

training_modifiers:
  - !EpochRangeModifier
    start_epoch: 0
    end_epoch: eval(num_epochs)

  - !LearningRateFunctionModifier
    start_epoch: eval(num_epochs - quantization_epochs)
    end_epoch: eval(num_epochs)
    lr_func: cosine
    init_lr: eval(quantization_lr)
    final_lr: eval(final_lr)

pruning_modifiers:
  - !ConstantPruningModifier
    start_epoch: 0.0
    params: __ALL_PRUNABLE__

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

The "Modifiers" encode how SparseML should modify the training process for Sparse Transfer Learning.
- `ConstantPruningModifier` tells SparseML to pin weights at 0 over all epochs, maintaining the sparsity structure of the network
- `QuantizationModifier` tells SparseML to quanitze the weights with quantization aware training over the last 5 epochs

SparseML parses the instructions declared in the recipe, and modifies the YOLOv5 training loop accordingly.

As a result, when the script completes, you will have a 75% pruned-quantized version of YOLOv5s trained on your data!

### Exporting for Inference

Once trained, you can export the model to ONNX for inference with DeepSparse. 

Run the following:

```bash 
!sparseml.yolov5.export_onnx \
  --weights yolov5_runs/train/exp/weights/last.pt \
  --dynamic
```

The resulting ONNX file is saved in your local directory.

## Other YOLOv5 Models

Here are some sample transfer learning commands for other versions of YOLOv5. Checkout the [SparseZoo](https://sparsezoo.neuralmagic.com/?page=1&domain=cv&sub_domain=detection) for the full repository of pre-sparsified checkpoints.

   - YOLOv5n Pruned-Quantized:
     ```bash
     sparseml.yolov5.train \
      --cfg yolov5n.yaml \
      --weights zoo:cv/detection/yolov5-n/pytorch/ultralytics/coco/pruned40_quant-none?recipe_type=transfer_learn \
      --recipe zoo:cv/detection/yolov5-n/pytorch/ultralytics/coco/pruned40_quant-none?recipe_type=transfer_learn \
      --data VOC.yaml \
      --patience 0 \
      --hyp hyps/hyp.finetune.yaml
     ```
   - YOLOv5s Pruned-Quantized:
     ```bash
     sparseml.yolov5.train \
        --cfg yolov5s.yaml \
        --weights zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned75_quant-none?recipe_type=transfer_learn \
        --recipe zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned75_quant-none?recipe_type=transfer_learn \
        --data VOC.yaml \
        --patience 0 \
        --hyp hyps/hyp.finetune.yaml
     ```
   - YOLOv5m Pruned-Quantized:
     ```bash
     sparseml.yolov5.train \
        --cfg yolov5m.yaml \
        --weights zoo:cv/detection/yolov5-m/pytorch/ultralytics/coco/pruned70_quant-none?recipe_type=transfer_learn \
        --recipe zoo:cv/detection/yolov5-m/pytorch/ultralytics/coco/pruned70_quant-none?recipe_type=transfer_learn \
        --data VOC.yaml \
        --patience 0 \
        --hyp hyps/hyp.finetune.yaml
     ```
   - YOLOv5l Pruned-Quantized:
     ```bash
      sparseml.yolov5.train \
        --cfg yolov5l.yaml \
        --weights zoo:cv/detection/yolov5-l/pytorch/ultralytics/coco/pruned90_quant-none?recipe_type=transfer_learn \
        --recipe zoo:cv/detection/yolov5-l/pytorch/ultralytics/coco/pruned90_quant-none?recipe_type=transfer_learn \
        --data VOC.yaml \
        --patience 0 \
        --hyp hyps/hyp.finetune.yaml
     ```
   - YOLOv5x Pruned-Quantized
     ```bash
     sparseml.yolov5.train \
        --cfg yolov5x.yaml \
        --weights zoo:cv/detection/yolov5-x/pytorch/ultralytics/coco/pruned80_quant-none?recipe_type=transfer_learn \
        --recipe zoo:cv/detection/yolov5-x/pytorch/ultralytics/coco/pruned80_quant-none?recipe_type=transfer_learn \
        --data VOC.yaml \
        --patience 0 \
        --hyp hyps/hyp.finetune.yaml
     ```

## Using a Custom Dataset

Because SparseML is integrated with Ultralytics, we can easily pass custom datasets to the training flows.

### Dataset Format 

There are three steps to creating a custom dataset for YOLOv5.

#### 1. Create `dataset.yaml`

Ultralytics uses a YAML file to pass a dataset configuration that defines:
- The dataset root directory `path` and relative paths to `train` / `val` / `test` image directories (or *.txt files with image paths)
- A class names dictionary

Here is an example for [COCO128](https://www.kaggle.com/datasets/ultralytics/coco128),
an example small tutorial dataset composed of the first 128 images in COCO train2017. These same 128 images are used for both training and validation to verify our training pipeline is capable of overfitting.

```yaml
# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: ../datasets/coco128  # dataset root dir
train: images/train2017  # train images (relative to 'path') 128 images
val: images/train2017  # val images (relative to 'path') 128 images
test:  # test images (optional)

# Classes (80 COCO classes)
names:
  0: person
  1: bicycle
  2: car
  ...
  77: teddy bear
  78: hair drier
  79: toothbrush
```

#### 2. Create Labels

After using a tool like [Roboflow Annotate](https://roboflow.com/annotate?ref=ultralytics) to label your data, export your labels to the YOLO Format, with one
`*.txt` file per image (if not objects are in the image, no `*.txt` file is require).

The `*.txt` file specifications are:
- One row per object
- Each row is `class x_center y_center width height` format.
- Box coordinates must be in normalized xywh format (from 0 - 1). If your boxes are in pixels, divide `x_center` and `width` by image width, and `y_center` and `height` by image height.
- Class numbers are zero-indexed (start from 0).


#### 3. Organize Directories 

Organize your train and val images and labels according to the example below. For the demo COCO128 file above, YOLOv5 assumes `/coco128` is inside a `/datasets` directory next to the `/yolov5` directory. YOLOv5 locates labels automatically for each image by replacing the last instance of `/images/` in each image path with `/labels/`. For example:

```
../datasets/coco128/images/im0.jpg  # image
../datasets/coco128/labels/im0.txt  # label
```

For more details, checkout the [custom dataset set tutorial](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data) in the Ultralytics repository.

### Example

Let's try a real example with an aerial dataset.

#### Download the Dataset

Download the dataset from Google Drive:

TO BE UPDATED:
```python
from google.colab import drive
drive.mount('/content/drive')
%cp /content/drive/MyDrive/aerial-dataset.tar.gz /content/
!tar -xf aerial-dataset.tar.gz
```

We can see that the dataset conforms to the Ultralytics format. After unzipping the directory looks as follows:

```
|-- aerial-dataset
  |--train
    |--images
      |--00001_frame000000_original.jpg
      ...
    |--labels
      |--00001_frame000000_original.txt
      ...
  |--val
    |--images
      |--00053_frame000000_original.jpg
      ...
    |--labels
      |--00053_frame000000_original.txt
      ...
```

Here is a sample label file for `aerial-dataset/train/labels/00001_frame000000_original.txt`:
```
0 0.719010 0.124074 0.022396 0.083333
0 0.943229 0.133333 0.039583 0.037037
0 0.787240 0.153241 0.042188 0.041667
0 0.741667 0.121759 0.017708 0.073148
0 0.693229 0.100463 0.017708 0.063889
0 0.670312 0.097222 0.025000 0.075926
0 0.648177 0.077315 0.022396 0.069444
0 0.619531 0.050463 0.022396 0.056481
0 0.492448 0.078704 0.039062 0.059259
0 0.418229 0.806019 0.034375 0.065741
0 0.349479 0.646296 0.018750 0.064815
0 0.461458 0.916204 0.037500 0.052778
```

Run the following to visualize:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random, cv2
from PIL import Image

plt.figure(figsize=(15, 15), facecolor='white')

SCENE = random.choice(val_scenes)
FRAME = df[df['scene'] == SCENE].sample()["frame"].iloc[0]

filename_image = f"aerial-dataset/val/images/00001_frame000000_original.jpg"
filename_label = filename_image.replace('images','labels').replace('jpg','txt')
data = pd.read_csv(filename_label, header=None, delimiter=' ', names=["class", "x_center", "y_center", "width", "height"])

print(filename_image)
print(filename_label)
im = cv2.imread(filename_image)
im_size = im.shape[:2]
for _, bbox in data.iterrows():

  cls, xc, yc, w, h = bbox
  xmin = xc - w/2
  ymin = yc - h/2
  xmax = xc + w/2
  ymax = yc + h/2

  xmin *= im_size[1]
  ymin *= im_size[0]
  xmax *= im_size[1]
  ymax *= im_size[0]

  start_point = (int(xmin), int(ymin))
  end_point = (int(xmax), int(ymax))
  color = (0, 255, 0)
  thickness = 2

  im = cv2.rectangle(im, start_point, end_point, color, thickness)

plt.axis("off")
plt.imshow(im)
```

#### Create a Config

Save the following configuration file as `aerial-dataset.yaml`:

```yaml
# aerial-dataset.yaml
path: /content/aerial-dataset
train:
  - train/images
val:
  - val/images

# Classes
nc: 1  # number of classes
names: ['object']
```

#### Run Transfer Learning

With the config file setup and data downloaded, we can simply swap in the dataset configuration file in place of the `VOC.yaml`.

```
sparseml.yolov5.train \
  --weights zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned75_quant-none?recipe_type=transfer_learn \
  --recipe zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned75_quant-none?recipe_type=transfer_learn \
  --recipe_args '{"num_epochs":30}'
  --data aerial-daraset.yaml.yaml \
  --patience 0 \
  --cfg yolov5s.yaml \
  --hyp hyps/hyp.finetune.yaml
```

You will notice that we added a `--recipe_args` argument, which updates the transfer 
learning recipe to run for 30 epochs rather than 55 epochs. While you can always create
a custom recipe file and pass a local file to script, the `--recipe_args` enables you
to modify on the fly.

## Next Steps

Checkout the DeepSparse repository for more information on deploying your sparse YOLOv5 models with DeepSparse for GPU class performance on CPUs.
