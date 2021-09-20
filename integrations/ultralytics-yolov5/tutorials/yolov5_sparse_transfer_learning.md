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

This tutorial shows how Neural Magic sparse models simplify the sparsification process by offering pre-sparsified YOLOv5 models for transfer learning onto other datasets.

## Overview

Neural Magic’s ML team creates sparsified models that allow anyone to plug in their data and leverage pre-sparsified models from the SparseZoo on top of Ultralytics’ robust training pipelines.
Sparsifying involves removing redundant information from neural networks using algorithms such as pruning and quantization, among others.
This sparsification process results in many benefits for deployment environments, including faster inference and smaller file sizes.
Unfortunately, many have not realized the benefits due to the complicated process and number of hyperparameters involved.
Working through this tutorial, you will experience how Neural Magic recipes simplify the sparsification process by:

- Downloading and preparing a pre-sparsified model.
- Setting up your custom data.
- Applying a sparse transfer learning recipe.
- Exporting an inference graph to reduce its deployment size and run it in an inference engine such as DeepSparse.

The examples listed in this tutorial are all performed on the VOC dataset.
Additionally, the results listed in this tutorial are available publicly through a [Weights and Biases project](https://wandb.ai/neuralmagic/yolov5-voc-sparse-transfer-learning).

Before diving in, be sure to go through setup as listed out in the [README](https://github.com/neuralmagic/sparseml/blob/main/integrations/ultralytics-yolov5/README.md) for this integration.
Additionally, all commands are intended to be run from the root of the `yolov5` repository folder (`cd integrations/ultralytics-yolov5/yolov5`).

## Need Help?

For Neural Magic Support, sign up or log in to get help with your questions in our Tutorials channel: [Discourse Forum](https://discuss.neuralmagic.com/) and/or [Slack](https://join.slack.com/t/discuss-neuralmagic/shared_invite/zt-q1a1cnvo-YBoICSIw3L1dmQpjBeDurQ).

## Selecting a Pre-sparsified Model

1. To begin transfer learning, first select from one of the sparsified models available through the [SparseZoo](https://sparsezoo.neuralmagic.com/?repo=ultralytics&page=1).
   These models were originally pruned on the COCO dataset achieving the following metrics:

   | Sparsification Type      | Description                                                                               | COCO mAP@0.5 | Size on Disk | DeepSparse Performance\*\* |
   | ------------------------ | ----------------------------------------------------------------------------------------- | ------------ | ------------ | -------------------------- |
   | YOLOv5s Baseline         | The baseline, small YOLOv5 model used as the starting point for sparsification.           | 0.556        | 24.8 MB      | 78.2 img/sec               |
   | YOLOv5s Pruned           | A highly sparse, FP32 YOLOv5s model that recovers close to the baseline model.            | 0.534        | 8.4 MB       | 100.5 img/sec              |
   | YOLOv5s Pruned Quantized | A highly sparse, INT8 YOLOv5s model that recovers reasonably close to the baseline model. | 0.525        | 3.3 MB       | 198.2 img/sec              |
   | YOLOv5l Baseline         | The baseline, large YOLOv5 model used as the starting point for sparsification.           | 0.654        | 154 MB       | 22.7 img/sec               |
   | YOLOv5l Pruned           | A highly sparse, FP32 YOLOv5l model that recovers close to the baseline model.            | 0.643        | 32.8 MB      | 40.1 img/sec               |
   | YOLOv5l Pruned Quantized | A highly sparse, INT8 YOLOv5l model that recovers reasonably close to the baseline model. | 0.623        | 12.7 MB      | 98.6 img/sec               |

   \*\* DeepSparse Performance measured on an AWS C5 instance with 24 cores, batch size 64, and 640x640 input with version 1.6 of the DeepSparse Engine.

2. After deciding on which model meets your performance requirements for both speed and accuracy, select the SparseZoo stub associated with that model.
   The stub will be used later with the training script to automatically pull down the desired pre-trained weights.
   - YOLOv5s Baseline: `zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/base-none`
   - YOLOv5s Pruned: `zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned-aggressive_96`
   - YOLOv5s Pruned Quantized: `zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned_quant-aggressive_94`
   - YOLOv5l Baseline: `zoo:cv/detection/yolov5-l/pytorch/ultralytics/coco/base-none`
   - YOLOv5l Pruned: `zoo:cv/detection/yolov5-l/pytorch/ultralytics/coco/pruned-aggressive_98`
   - YOLOv5l Pruned Quantized: `zoo:cv/detection/yolov5-l/pytorch/ultralytics/coco/pruned_quant-aggressive_95`

You are now ready to set up the data for training.

## Setting Up the Data

Note: If using your custom data, the Ultralytics repo contains a walk-through for [training custom data](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data).
Otherwise, setup scripts for both [VOC](https://cs.stanford.edu/~roozbeh/pascal-context/) and [COCO](https://cocodataset.org/#home) can be found under the [yolov5/data/scripts path](https://github.com/ultralytics/yolov5/tree/master/data/scripts).

1. For this tutorial, run the VOC setup script with the following command from the root of the `yolov5` repository:
   ```bash
   bash data/scripts/get_voc.sh
   ```
2. Download and validation of the VOC dataset will begin and take around 10 minutes to finish.
   The script downloads the VOC dataset into a `VOC` folder under the parent directory.
   Notice that, once completed, the data is ready for training with the folder structure in the following state:
   ```
   |-- VOC
   |   |-- images
   |   |   |-- train
   |   |   `-- val
   |   `-- labels
   |       |-- train
   |       `-- val
   |-- yolov5
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

You are ready to transfer learn the model.

## Transfer Learning the Model

The training command will take a few hours to complete (anywhere from 3 hours for YOLOV5s to 12 hours for YOLOv5l on an A100).
Afterward, you will have a sparse model transfer learned onto the VOC dataset.

The command uses the `--recipe` argument to encode the proper hyperparams such that SparseML will enforce the sparsity for all layers.
Without the proper recipes, the zeroed weights will slowly become dense as the model is trained further.
The recipes are specific to the sparsification type, so the training command will differ based on if you are transfer learning the pruned quantized, pruned, or baseline model.

1. Select the proper command to run based on the model and the sparsification type of the model you chose earlier.

   - YOLOv5s Pruned transfer learning:
     ```bash
     python train.py --data voc.yaml --cfg ../models/yolov5s.yaml --weights zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned-aggressive_96?recipe_type=transfer --hyp data/hyp.finetune.yaml --recipe ../recipes/yolov5.transfer_learn_pruned.md
     ```
   - YOLOv5s Pruned-Quantized transfer learning:
     ```bash
     python train.py --data voc.yaml --cfg ../models/yolov5s.yaml --weights zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned_quant-aggressive_94?recipe_type=transfer --hyp data/hyp.finetune.yaml --recipe ../recipes/yolov5.transfer_learn_pruned_quantized.md
     ```
   - YOLOv5s Baseline transfer learning:
     ```bash
     python train.py --data voc.yaml --cfg ../models/yolov5s.yaml --weights zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/base-none --hyp data/hyp.finetune.yaml --epochs 50
     ```
   - YOLOv5l Pruned transfer learning:
     ```bash
     python train.py --data voc.yaml --cfg ../models/yolov5l.yaml --weights zoo:cv/detection/yolov5-l/pytorch/ultralytics/coco/pruned-aggressive_98?recipe_type=transfer --hyp data/hyp.finetune.yaml --recipe ../recipes/yolov5.transfer_learn_pruned.md
     ```
   - YOLOv5l Pruned-Quantized transfer learning:
     ```bash
     python train.py --data voc.yaml --cfg ../models/yolov5l.yaml --weights zoo:cv/detection/yolov5-l/pytorch/ultralytics/coco/pruned_quant-aggressive_95?recipe_type=transfer --hyp data/hyp.finetune.yaml --recipe ../recipes/yolov5.transfer_learn_pruned_quantized.md
     ```
   - YOLOv5l Baseline transfer learning:
     ```bash
     python train.py --data voc.yaml --cfg ../models/yolov5l.yaml --weights zoo:cv/detection/yolov5-l/pytorch/ultralytics/coco/base-none --hyp data/hyp.finetune.yaml --epochs 50
     ```

   **_Notes About Transfer Learning_**

   - Check your CPU hardware support for quantized networks (VNNI instruction set) using the DeepSparse API:
     ```python
     from deepsparse.cpu import cpu_vnni_compatible
     print(f"VNNI available: {cpu_vnni_compatible()}")
     ```
   - If your hardware does not support quantized networks for inference speedup or complete recovery is very important, then Neural Magic recommends using the `pruned` recipe. The recipe to use depends on how long you are willing to train and how vital full recovery is. Consult the table above for this comparison.
   - If your hardware does support quantized networks, we recommend using the `pruned quantized` recipe. The recipe to use depends on how long you are willing to train and how crucial full recovery is. Consult the table for this comparison.
   - When running quantized models, the memory footprint for training will significantly increase (roughly 3x). It is recommended to train at a high batch size at first. This will fail with an out-of-memory exception once quantization starts. Once this happens, use the `last.pt` weights from that run to resume training with a lower batch size.

   **_VOC Results_**

   - Weights and Biases is very useful for comparing across the different runs; the resulting transfer learning graphs for epochs vs mAP@0.5 graph is supplied below for all of the options:

     <img src="https://raw.githubusercontent.com/neuralmagic/sparseml/main/integrations/ultralytics-yolov5/tutorials/images/transfer-learning-wandb-chart.png" width="960px" style="border: 2px solid #000000;" />

2. After training has finished, find the `best.pt` checkpoint to export for inference.
   The training command creates a `runs` directory under the `yolov5` repository directory.
   This directory will contain the outputs from the training run, including experimental results along with the trained model:
   ```
   |-- VOC
   |-- data
   |-- models
   |-- recipes
   |-- tutorials
   |-- yolov5
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

You are ready to export for inference.

## Exporting for Inference

This step loads a checkpoint file of the best weights measured on the validation set, and converts it into the more common inference formats.
Then, you can run the file through a compression algorithm to reduce its deployment size and run it in an inference engine such as DeepSparse.

The `best.pt` file, located in the previous step, contains a checkpoint of the best weights measured on the validation set.
These weights can be loaded into the `train.py` and `test.py` scripts now.
However, other formats are generally more friendly for other inference deployment platforms, such as [ONNX](https://onnx.ai/).

The [export.py script](https://github.com/neuralmagic/yolov5/blob/master/models/export.py) handles the logic behind loading the checkpoint and converting it into the more common inference formats, as described here.

1. Enter the following command to load the PyTorch graph, convert to ONNX, and correct any misformatted pieces of the graph for the pruned and quantized models.
   ```bash
   python models/export.py --weights PATH_TO_SPARSIFIED_WEIGHTS  --dynamic
   ```
   The result is a new file added next to the sparsified checkpoint with a `.onnx` extension:
   ```
   |-- exp
   |   |-- weights
   |   |   |-- best.pt
   |   |   |-- best.onnx
   |   |   `-- last.pt
   `-- ...
   ```
2. Now you can run the `.onnx` file through a compression algorithm to reduce its deployment size and run it in ONNX-compatible inference engines such as [DeepSparse](https://github.com/neuralmagic/deepsparse).
   The DeepSparse Engine is explicitly coded to support running sparsified models for significant improvements in inference performance.
   An example for benchmarking and deploying YOLOv5 models with DeepSparse can be found [here](https://github.com/neuralmagic/deepsparse/tree/main/examples/ultralytics-yolo).

## Wrap-Up

Neural Magic sparse models and recipes simplify the sparsification process by enabling sparse transfer learning to create highly accurate pruned and pruned-quantized YOLOv5 models.
In this tutorial, you downloaded a pre-sparsified model, applied a Neural Magic recipe for sparse transfer learning, and exported to ONNX to run through an inference engine.

Now, refer [here](https://github.com/neuralmagic/deepsparse/tree/main/examples/ultralytics-yolo) for an example for benchmarking and deploying YOLOv5 models with DeepSparse.

For Neural Magic Support, sign up or log in to get help with your questions in our Tutorials channel: [Discourse Forum](https://discuss.neuralmagic.com/) and/or [Slack](https://join.slack.com/t/discuss-neuralmagic/shared_invite/zt-q1a1cnvo-YBoICSIw3L1dmQpjBeDurQ).
