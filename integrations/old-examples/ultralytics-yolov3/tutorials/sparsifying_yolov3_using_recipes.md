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

# Sparsifying YOLOv3 Using Recipes

This tutorial shows how Neural Magic recipes simplify the sparsification process by encoding the hyperparameters and instructions needed to create highly accurate pruned and pruned-quantized YOLOv3 models.

## Overview

Neural Magic’s ML team creates recipes that allow anyone to plug in their data and leverage SparseML’s recipe-driven approach on top of Ultralytics’ robust training pipelines. Sparsifying involves removing redundant information from neural networks using algorithms such as pruning and quantization, among others. This sparsification process results in many benefits for deployment environments, including faster inference and smaller file sizes. Unfortunately, many have not realized the benefits due to the complicated process and number of hyperparameters involved.

Working through this tutorial, you will experience how Neural Magic recipes simplify the sparsification process by:

* Creating a pre-trained model to establish a baseline. You will set up your custom data and then train the model.
* Applying a recipe to select the trade off between the amount of recovery to the baseline training performance with the amount of sparsification for inference performance.
* Exporting for inference to run a file (that contains a checkpoint of the best weights measured on the validation set) through a compression algorithm to reduce its deployment size and run it in an inference engine such as [DeepSparse](https://github.com/neuralmagic/deepsparse).

The examples listed in this tutorial are all performed on the VOC dataset. Additionally, the results listed in this tutorial are available publicly through a [Weights and Biases project](https://wandb.ai/neuralmagic/yolov3-spp-lrelu-voc).

<div style="width: 100%; display: flex; justify-content: center;">
    <a href="https://youtu.be/o5qIYs47MPw" target="_blank">
        <img alt="Example YOLOv3 Inference Video" src="https://raw.githubusercontent.com/neuralmagic/sparseml/main/integrations/ultralytics-yolov3/tutorials/images/pruned-quantized-result.jpeg" width="560px" style="border: 2px solid #000000;" />
    </a>
</div>

## Need Help?

For Neural Magic Support, sign up or log in to our [**Deep Sparse Community Slack**](https://join.slack.com/t/discuss-neuralmagic/shared_invite/zt-q1a1cnvo-YBoICSIw3L1dmQpjBeDurQ). Bugs, feature requests, or additional questions can also be posted to our [GitHub Issue Queue.](https://github.com/neuralmagic/sparseml/issues)

## Creating a Pre-trained Model

Before applying one of the recipes, you must first create the pre-trained model to sparsify further. The pre-trained model enables pruning and other algorithms to remove the correct redundant information in place of random information. Your goal after this is to create a smaller, faster model that recovers to the pre-trained baseline.

Creating a pre-trained model involves two steps: 1) setting up the data and 2) training the model.

**Note**: If using your custom data, the Ultralytics repo contains a walk-through for [training custom data](https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data). Otherwise, setup scripts for both [VOC](https://cs.stanford.edu/~roozbeh/pascal-context/) and [COCO](https://cocodataset.org/#home) can be found under the [yolov3/data/scripts path](https://github.com/ultralytics/yolov3/tree/master/data/scripts).


### Setting Up the Data

1. For this tutorial, we run the VOC setup script with the following command from the root of the yolov3 repository:
```bash
bash data/scripts/get_voc.sh
```
2. Download and validation of the VOC dataset will begin and takes around 10 minutes to finish.
The script downloads the VOC dataset into a `VOC` folder under the parent directory.
Notie that, once completed, the data is ready for training with the folder structure in the following state:
```
|-- VOC
|   |-- images
|   |   |-- train
|   |   `-- val
|   `-- labels
|       |-- train
|       `-- val
|-- yolov3
|   |-- data
|   |-- models
|   |-- utils
|   |-- weights
|   |-- Dockerfile
|   |-- LICENSE
|   |-- README.md
|   |-- detect.py
|   |-- hubconf.py
|   |-- requirements.txt
|   |-- test.py
|   |-- train.py
|   `-- tutorial.ipynb
```

You are ready to train the model.

### Training the Model

The training command will take a few hours to complete (~3 hours for a Titan RTX). Afterward, you will have a model that achieves roughly 0.85 mAP on the VOC dataset ready for sparsifying.

1. To expedite the training process, you will transfer learn from weights initially trained on the COCO dataset. These are stored in the SparseZoo and accessed with the following Python code. Enter:

```python
from sparsezoo import Model

model = Model("zoo:cv/detection/yolo_v3-spp/pytorch/ultralytics/coco/base-none")
checkpoint_path = model.training.default.path
print(checkpoint_path)
```

After running, the code has downloaded the checkpoint file to the local system:
```bash
downloading...: 100%|██████████| 126342273/126342273 [00:04<00:00, 28306365.97it/s]
/Users/markkurtz/.cache/sparsezoo/16fe8358-2e91-4b81-a1f2-df85bd1a9ac3/pytorch/model.pth
```

2. Next, the checkpoint file provides the source weights for training on VOC. Run the following train command from within the yolov3 repository folder:
```bash
python train.py --weights PATH_TO_COCO_PRETRAINED_WEIGHTS --data voc.yaml --img 512 --epochs 50
```

The training command creates a `runs` directory under the yolov3 repository directory.
This directory will contain the outputs from the training run, including experimental results along with the trained model:
```
|-- VOC
|-- data
|-- models
|-- recipes
|-- tutorials
|-- yolov3
|   |-- data
|   |-- models
|   |-- runs
|   |   `-- train
|   |       |-- exp
|   |       |   |-- weights
|   |       |   |   |-- best.pt
|   |       |   |   `-- last.pt
|   |       |   |-- F1_curve.png
|   |       |   |-- PR_curve.png
|   |       |   |-- P_curve.png
|   |       |   |-- R_curve.png
|   |       |   |-- confusion_matrix.png
|   |       |   `-- ...
|   |-- train.py
|   `-- ...
|-- README.md
`-- setup_integration.sh
```

You are ready to use the weights at `yolov3/runs/train/exp/weights/best.pt` with the Neural Magic recipes to create a sparsified model.

## Applying a Recipe

In general, recipes trade off the amount of recovery to the baseline training performance with the amount of sparsification for inference performance.
The [`recipes` folder](https://github.com/neuralmagic/sparseml/blob/main/integrations/old-examples/ultralytics-yolov3/recipes) contains multiple files, each offering certain advantages over others. The table below compares these tradeoffs and shows how to run them on the VOC dataset.
1. Review this table, which lists recipes, commands, and results.

| Recipe Name                                                                                                                                              | Description                                                                                                                     | Train Command                                                                                                                                                                               | VOC mAP@0.5 | Size on Disk | DeepSparse Performance** | Training Epochs (time) | Weights and Biases                                                       |
|----------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------|--------------|--------------------------|------------------------|--------------------------------------------------------------------------|
| Baseline                                                                                                                                                 | The baseline, pretrained model originally transfer learned onto the VOC dataset.                                                | ``` python train.py --weights PATH_TO_COCO_PRETRAINED_WEIGHTS --data voc.yaml --img 512 --epochs 50 ```                                                                                     | 0.857       | 194 MB       | 19.5 img/sec             | 50 (3.21 hours)        | [wandb](https://wandb.ai/neuralmagic/yolov3-spp-lrelu-voc/runs/261llnb1) |
| [Pruned](https://github.com/neuralmagic/sparseml/blob/main/integrations/old-examples/ultralytics-yolov3/recipes/yolov3-spp.pruned.md)                                 | Creates a highly sparse, FP32 model that recovers close to the baseline model.                                                  | ``` python train.py --weights PATH_TO_VOC_PRETRAINED_WEIGHTS --data voc.yaml --img 512 --hyp ../data/hyp.pruned.yaml --recipe ../recipes/yolov3-spp.pruned.md ```                           | 0.858       | 33.6 MB      | 34.0 img/sec             | 300 (20.45 hours)      | [wandb](https://wandb.ai/neuralmagic/yolov3-spp-lrelu-voc/runs/2jeadrts) |
| [Pruned Short](https://github.com/neuralmagic/sparseml/blob/main/integrations/old-examples/ultralytics-yolov3/recipes/yolov3-spp.pruned.short.md)                     | Creates a highly sparse, FP32 model in a shortened schedule to prioritize quicker training while sacrificing a bit on recovery. | ``` python train.py --weights PATH_TO_VOC_PRETRAINED_WEIGHTS --data voc.yaml --img 512 --hyp ../data/hyp.pruned.yaml --recipe ../recipes/yolov3-spp.pruned.short.md ```                     | 0.854       | 33.6 MB      | 34.0 img/sec             | 80 (5.53 hours)        | [wandb](https://wandb.ai/neuralmagic/yolov3-spp-lrelu-voc/runs/jktw650n) |
| [Pruned Quantized](https://github.com/neuralmagic/sparseml/blob/main/integrations/old-examples/ultralytics-yolov3/recipes/yolov3-spp.pruned_quantized.md)             | Creates a highly sparse, INT8 model that recovers reasonably close to the baseline model.                                       | ``` python train.py --weights PATH_TO_VOC_PRETRAINED_WEIGHTS --data voc.yaml --img 512 --hyp ../data/hyp.pruned_quantized.yaml --recipe ../recipes/yolov3-spp.pruned_quantized.md ```       | 0.827       | 13.4 MB      | 90.4 img/sec             | 242 (18.32 hours)      | [wandb](https://wandb.ai/neuralmagic/yolov3-spp-lrelu-voc/runs/2dfy3rgs) |
| [Pruned Quantized Short](https://github.com/neuralmagic/sparseml/blob/main/integrations/old-examples/ultralytics-yolov3/recipes/yolov3-spp.pruned_quantized.short.md) | Creates a highly sparse, INT8 model in a shortened schedule to prioritize quicker training while sacrificing a bit on recovery. | ``` python train.py --weights PATH_TO_VOC_PRETRAINED_WEIGHTS --data voc.yaml --img 512 --hyp ../data/hyp.pruned_quantized.yaml --recipe ../recipes/yolov3-spp.pruned_quantized.short.md ``` | 0.808       | 13.4 MB      | 90.4 img/sec             | 52 (4.23 hours)        | [wandb](https://wandb.ai/neuralmagic/yolov3-spp-lrelu-voc/runs/1picfimy) |
| [Test](https://github.com/neuralmagic/sparseml/blob/main/integrations/old-examples/ultralytics-yolov3/recipes/yolov3-spp.test.md)                                     | A test recipe to test the training pipeline and device for both pruning and quantization in 5 epochs.                           | ``` python train.py --weights PATH_TO_VOC_PRETRAINED_WEIGHTS --data voc.yaml --img 512 --hyp ../data/hyp.pruned_quantized.yaml --recipe ../recipes/yolov3-spp.test.md ```                   | 0.702       | 13.4 MB      | 90.4 img/sec             | 5 (17 minutes)         | [wandb](https://wandb.ai/neuralmagic/yolov3-spp-lrelu-voc/runs/3bkw6c60) |

** DeepSparse Performance measured on an AWS C5 instance with 24 cores, batch size 64, and 640x640 input with version 1.3 of the DeepSparse Engine.

2. Notice the Weights and Biases information in the table, which is very useful for comparing across these runs. The epochs versus mAP@0.5 graph is:

<img src="https://raw.githubusercontent.com/neuralmagic/sparseml/main/integrations/ultralytics-yolov3/tutorials/images/pruned-quantized-wandb-chart.png" width="960px" style="border: 2px solid #000000;" />

**Notes About Recipe Selection**

If your hardware does not support quantized networks for inference speedup or complete recovery is very important, then Neural Magic recommends using either the  `pruned` or `pruned short` recipe. The recipe to use depends on how long you are willing to train and how vital full recovery is. Consult the table above for this comparison.

If your hardware does support quantized networks (VNNI instruction set on CPUs, for example), we recommend using the `pruned quantized` or `pruned quantized short` recipe. The recipe to use depends on how long you are willing to train and how crucial full recovery is. Consult the table for this comparison.

When running quantized models, the memory footprint for training will significantly increase (roughly 3x). Therefore it is essential to take this into account when selecting a batch size to train at. To ensure no issues with the longer quantized runs, run the quicker test recipe first to ensure your configurations are correct and the training process will complete successfully.

3. To begin applying one of the recipes, use the `--recipe` argument within the Ultralytics [train script](https://github.com/neuralmagic/yolov3/blob/master/train.py).
In addition, the hyperparameters are changed slightly to better work with the recipe.
These hyperparameters are stored in appropriately named files under the [`data` directory](https://github.com/neuralmagic/sparseml/blob/main/integrations/old-examples/ultralytics-yolov3/data) and are passed into the training script using the `--hyp` argument.
Both of the arguments are combined with our previous training command and VOC pre-trained weights to run the recipes over the model. For example:
```bash
python train.py --weights PATH_TO_COCO_PRETRAINED_WEIGHTS --data voc.yaml --img 512 --epochs 50 --recipe PATH_TO_SPARSIFICATION_RECIPE
```
After applying a recipe, you are ready to export for inference.

## Exporting for Inference

This step loads a checkpoint file of the best weights measured on the validation set, and converts it into the more common inference formats. Then, you can run the file through a compression algorithm to reduce its deployment size and run it in an inference engine such as [DeepSparse](https://github.com/neuralmagic/deepsparse).

When you applied a recipe in the previous step, the sparsification run created a new `exp` directory under the yolov3 runs directory:

```
|-- VOC
|-- data
|-- models
|-- recipes
|-- tutorials
|-- yolov3
|   |-- data
|   |-- models
|   |-- runs
|   |   `-- train
|   |       |-- exp
|   |       |-- exp1
|   |       |   |-- weights
|   |       |   |   |-- best.pt
|   |       |   |   `-- last.pt
|   |       |   |-- F1_curve.png
|   |       |   |-- PR_curve.png
|   |       |   |-- P_curve.png
|   |       |   |-- R_curve.png
|   |       |   |-- confusion_matrix.png
|   |       |   `-- ...
|   |-- train.py
|   `-- ...
|-- README.md
`-- setup_integration.sh
```

The `best.pt` file contains a checkpoint of the best weights measured on the validation set.
These weights can be loaded into the `train.py` and `test.py` scripts now. However, other formats are generally more friendly for other inference deployment platforms, such as [ONNX](https://onnx.ai/).

The [`export.py` script](https://github.com/neuralmagic/yolov3/blob/master/models/export.py) handles the logic behind loading the checkpoint and converting it into the more common inference formats, as described here.

1. Enter the following command to load the PyTorch graph, convert to ONNX, and then correct any misformatted pieces for the pruned and quantized models.

```bash
python models/export.py --weights PATH_TO_SPARSIFIED_WEIGHTS --img-size 512 512
```

The result is a new file added next to the sparsified checkpoint with a `.onnx` extension:
```
|-- exp
|-- exp1
|-- weights
|   |-- best.pt
|   |-- best.onnx
|   `-- last.pt
`-- ...
```

2. Now you can run the `.onnx` file through a compression algorithm to reduce its deployment size and run it in ONNX-compatible inference engines such as [DeepSparse](https://github.com/neuralmagic/deepsparse).
 
The DeepSparse Engine is explicitly coded to support running sparsified models for significant improvements in inference performance. An example for benchmarking and deploying YOLOv3 models with DeepSparse can be found [here](https://github.com/neuralmagic/deepsparse/tree/main/examples/ultralytics-yolo).

## Wrap-Up

Neural Magic recipes simplify the sparsification process by encoding the hyperparameters and instructions needed to create highly accurate pruned and pruned-quantized YOLOv3 models. In this tutorial, you created a pre-trained model to establish a baseline, applied a Neural Magic recipe for sparsification, and exported to ONNX to run through an inference engine.

Now, refer [here](https://github.com/neuralmagic/deepsparse/tree/main/examples/ultralytics-yolo) for an example for benchmarking and deploying YOLOv3 models with DeepSparse.

For Neural Magic Support, sign up or log in to our [**Deep Sparse Community Slack**](https://join.slack.com/t/discuss-neuralmagic/shared_invite/zt-q1a1cnvo-YBoICSIw3L1dmQpjBeDurQ). Bugs, feature requests, or additional questions can also be posted to our [GitHub Issue Queue.](https://github.com/neuralmagic/sparseml/issues)
