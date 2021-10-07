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

# Sparsifying YOLOv5 Using Recipes

This tutorial shows how Neural Magic recipes simplify the sparsification process by encoding the hyperparameters and instructions needed to create highly accurate pruned and pruned-quantized YOLOv5 models, specifically for the s and l versions.

## Overview

Neural Magic's ML team creates recipes that allow anyone to plug in their data and leverage SparseML's recipe-driven approach on top of Ultralytics' robust training pipelines.
Sparsifying involves removing redundant information from neural networks using algoriths such as pruning and quantization, among others.
This sparsification process results in many benefits for deployment environments, including faster inference and smaller file sizes.
Unfortunately, many have not realized the benefits due to the complicated process and number of hyperparameters involved.

Working through this tutorial, you will experience how Neural Magic recipes simplify the sparsification process by:

* Creating a pre-trained model to establish a baseline. You will set up your custom data and then train the model.
* Applying a recipe to select the trade off between the amount of recovery to the baseline training performance with the amount of sparsification for inference performance.
* Exporting for inference to run a file (that contains a checkpoint of the best weights measured on the validation set) through a compression algorithm to reduce its deployment size and run it in an inference engine such as [DeepSparse](https://github.com/neuralmagic/deepsparse).

The examples listed in this tutorial are all performed on the COCO dataset.

Before diving in, be sure to go through setup as listed out in the [README](https://github.com/neuralmagic/sparseml/blob/main/integrations/ultralytics-yolov5/README.md) for this integration.
Additionally, all commands are intended to be run from the root of the `yolov5` repository folder (`cd integrations/ultralytics-yolov5/yolov5`).

## Need Help?

For Neural Magic Support, sign up or log in to get help with your questions in our **Tutorials channel**: [Discourse Forum](https://discuss.neuralmagic.com/) and/or [Slack](https://join.slack.com/t/discuss-neuralmagic/shared_invite/zt-q1a1cnvo-YBoICSIw3L1dmQpjBeDurQ). 

## Creating a Pre-trained Model

Before applying one of the recipes, you must first create the pre-trained model to sparsify further. 
The pre-trained model enables pruning and other algorithms to remove the correct redundant information in place of random information. 
Your goal after this is to create a smaller, faster model that recovers to the pre-trained baseline.

Creating a pre-trained model involves two steps: 1) setting up the data and 2) training the model.

**Note**: If using your custom data, the Ultralytics repo contains a walk-through for [training custom data](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data). 
Otherwise, setup scripts for both [VOC](https://cs.stanford.edu/~roozbeh/pascal-context/) and [COCO](https://cocodataset.org/#home) can be found under the [yolov5/data/scripts path](https://github.com/ultralytics/yolov5/tree/master/data/scripts).

### Setting Up the Data

1. For this tutorial, run the COCO setup script with the following command from the root of the `yolov5` repository:
    ```bash
    bash data/scripts/get_coco.sh
    ```
2. Download and validation of the COCO dataset will begin and takes around 10 minutes to finish.
    The script downloads the COCO dataset into a `coco` folder under the parent directory.
    Notice that, once completed, the data is ready for training with the folder structure in the following state:
    ```
    |-- coco
    |   |-- annotations
    |   |-- images
    |   |-- labels
    |   |-- LICENSE
    |   |-- README.txt
    |   |-- test-dev2017.txt
    |   |-- train2017.cache
    |   |-- train2017.txt
    |   |-- val2017.cache
    |   `-- val2017.txt
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
   
    You are ready to train the model.

### Training the Model

The training command will take multiple hours to complete since it is training from scratch (anywhere from 12 hours for YOLOV5s to 48 hours for YOLOv5l on 4 A100's). 
Afterward, you will have a model that achieves roughly 0.556 mAP@0.5 or 0.654 mAP@0.5 on the COCO dataset for YOLOv5s and YOLOv5l respectively ready for sparsifying.

1. To create a deployable model, you will train a YOLOv5 version with HardSwish activations from scratch.
   To decide which version to use, consult the table below in the [Applying a Recipe](#applying-a-recipe) section. 
   The training commands for both are given below.
   
   YOLOv5s:
   ```bash
   python train.py --cfg ../models/yolov5s.yaml --weights "" --data coco.yaml --hyp data/hyp.scratch.yaml
   ```    

   YOLOv5l:
   ```bash
   python train.py --cfg ../models/yolov5l.yaml --weights "" --data coco.yaml --hyp data/hyp.scratch.yaml
   ```
2. Validate that the training commands completed successfully by checking under the newly created `runs/train/exp` path for the trained weights.
   The best trained weights will be found at `runs/train/exp/weights/best.pt` and will be used later for further sparsification.
   The full results are also visible in this folder under multiple formats including TensorBoard runs, txt files, and png files.
   Upon success, the results directory should look like the following:
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

You are ready to use the weights at `yolov5/runs/train/exp/weights/best.pt` with the Neural Magic recipes to create a sparsified model.

## Applying a Recipe

In general, recipes trade off the amount of recovery to the baseline training performance with the amount of sparsification for inference performance.
The [`recipes` folder](https://github.com/neuralmagic/sparseml/blob/main/integrations/ultralytics-yolov5/recipes) contains multiple files, each offering certain advantages over others. 
The table below compares these tradeoffs and shows how to run them on the COCO dataset.
1. Review this table, which lists recipes, commands, and results.

    | Recipe Name                                                                                                                                              | Description                                                                                                                     | Train Command                                                                                                                                                                               | COCO mAP@0.5 | Size on Disk | DeepSparse Performance** |
    |----------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------|--------------|--------------------------|
    | YOLOv5s Baseline                                                                                                                                         | The baseline, small YOLOv5 model used as the starting point for sparsification.                                                 | ``` python train.py --cfg ../models/yolov5s.yaml --weights "" --data coco.yaml --hyp data/hyp.scratch.yaml ```                                                                              | 0.556        | 24.8 MB       | 78.2 img/sec             |
    | [YOLOv5s Pruned](https://github.com/neuralmagic/sparseml/blob/main/integrations/ultralytics-yolov5/recipes/yolov5s.pruned.md)                            | Creates a highly sparse, FP32 YOLOv5s model that recovers close to the baseline model.                                          | ``` python train.py --cfg ../models/yolov5s.yaml --weights PATH_TO_COCO_PRETRAINED_WEIGHTS --data coco.yaml --hyp data/hyp.scratch.yaml --recipe ../recipes/yolov5s.pruned.md ```           | 0.534        | 8.4 MB      | 100.5 img/sec            |
    | [YOLOv5s Pruned Quantized](https://github.com/neuralmagic/sparseml/blob/main/integrations/ultralytics-yolov5/recipes/yolov5s.pruned_quantized.md)        | Creates a highly sparse, INT8 YOLOv5s model that recovers reasonably close to the baseline model.                               | ``` python train.py --cfg ../models/yolov5s.yaml --weights PATH_TO_COCO_PRETRAINED_WEIGHTS --data coco.yaml --hyp data/hyp.scratch.yaml --recipe ../recipes/yolov5s.pruned_quantized.md ``` | 0.525        | 3.3 MB      | 198.2 img/sec            |
    | YOLOv5l Baseline                                                                                                                                         | The baseline, large YOLOv5 model used as the starting point for sparsification.                                                 | ``` python train.py --cfg ../models/yolov5l.yaml --weights "" --data coco.yaml --hyp data/hyp.scratch.yaml ```                                                                              | 0.654        | 154 MB      | 22.7 img/sec             |
    | [YOLOv5l Pruned](https://github.com/neuralmagic/sparseml/blob/main/integrations/ultralytics-yolov5/recipes/yolov5l.pruned.md)                            | Creates a highly sparse, FP32 YOLOv5l model that recovers close to the baseline model.                                          | ``` python train.py --cfg ../models/yolov5l.yaml --weights PATH_TO_COCO_PRETRAINED_WEIGHTS --data coco.yaml --hyp data/hyp.scratch.yaml --recipe ../recipes/yolov5l.pruned.md ```           | 0.643        | 32.8 MB       | 40.1 img/sec             |
    | [YOLOv5l Pruned Quantized](https://github.com/neuralmagic/sparseml/blob/main/integrations/ultralytics-yolov5/recipes/yolov5l.pruned_quantized.md)        | Creates a highly sparse, INT8 YOLOv5l model that recovers reasonably close to the baseline model.                               | ``` python train.py --cfg ../models/yolov5l.yaml --weights PATH_TO_COCO_PRETRAINED_WEIGHTS --data coco.yaml --hyp data/hyp.scratch.yaml --recipe ../recipes/yolov5l.pruned_quantized.md ``` | 0.623        | 12.7 MB       | 98.6 img/sec             |

    ** DeepSparse Performance measured on an AWS C5 instance with 24 cores, batch size 64, and 640x640 input with version 1.6 of the DeepSparse Engine.

2. Select a recipe to use on top of the pre-trained model you created.

    ***Notes About Recipe Selection***
    - The recipes are not interchangeable between model versions. For example, if you created a pre-trained YOLOv5l model then be sure to use a recipe listed for v5l.
    - Check your CPU hardware support for quantized networks (VNNI instruction set) using the DeepSparse API:
      ```python
      from deepsparse.cpu import cpu_vnni_compatible
      print(f"VNNI available: {cpu_vnni_compatible()}")
      ```
    - If your hardware does not support quantized networks for inference speedup or complete recovery is very important, then Neural Magic recommends using the  `pruned` recipe. The recipe to use depends on how long you are willing to train and how vital full recovery is. Consult the table above for this comparison.
    - If your hardware does support quantized networks, we recommend using the `pruned quantized` recipe. The recipe to use depends on how long you are willing to train and how crucial full recovery is. Consult the table for this comparison.
    - When running quantized models, the memory footprint for training will significantly increase (roughly 3x). It is recommended to train at a high batch size at first. This will fail with an out-of-memory exception once quantization starts. Once this happens, use the `last.pt` weights from that run to resume training with a lower batch size.

3. To begin applying one of the recipes, use the `--recipe` argument within the Ultralytics [train script](https://github.com/neuralmagic/yolov5/blob/master/train.py).
   The recipe argument is combined with our previous training command and COCO pre-trained weights to run the recipes over the model. For example, a command for YOLOv5s would look like this:
   ```bash
   python train.py --cfg ../models/yolov5s.yaml --weights PATH_TO_COCO_PRETRAINED_WEIGHTS --data coco.yaml --hyp data/hyp.scratch.yaml --recipe PATH_TO_SPARSIFICATION_RECIPE
   ```
    After applying a recipe, you are ready to export for inference.

## Exporting for Inference

This step loads a checkpoint file of the best weights measured on the validation set, and converts it into the more common inference formats. 
Then, you can run the file through a compression algorithm to reduce its deployment size and run it in an inference engine such as [DeepSparse](https://github.com/neuralmagic/deepsparse).

When you applied a recipe in the previous step, the sparsification run created a new `exp` directory under the `yolov5/runs` directory:

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
These weights can be loaded into the `train.py` and `test.py` scripts now. 
However, other formats are generally more friendly for other inference deployment platforms, such as [ONNX](https://onnx.ai/).

The [`export.py` script](https://github.com/neuralmagic/yolov5/blob/master/models/export.py) handles the logic behind loading the checkpoint and converting it into the more common inference formats, as described here.

1. Enter the following command to load the PyTorch graph, convert to ONNX, and correct any misformatted pieces of the graph for the pruned and quantized models.

    ```bash
    python models/export.py --weights PATH_TO_SPARSIFIED_WEIGHTS --dynamic
    ```
    
    The result is a new file added next to the sparsified checkpoint with a `.onnx` extension:
    ```
    |-- exp
    |-- exp1
    |-- |-- weights
    |   |   |-- best.pt
    |   |   |-- best.onnx
    |   |   `-- last.pt
    `-- ...
    ```

2. Now you can run the `.onnx` file through a compression algorithm to reduce its deployment size and run it in ONNX-compatible inference engines such as [DeepSparse](https://github.com/neuralmagic/deepsparse).
   The DeepSparse Engine is explicitly coded to support running sparsified models for significant improvements in inference performance. 
   An example for benchmarking and deploying YOLOv5 models with DeepSparse can be found [here](https://github.com/neuralmagic/deepsparse/tree/main/examples/ultralytics-yolo).

## Wrap-Up

Neural Magic recipes simplify the sparsification process by encoding the hyperparameters and instructions needed to create highly accurate pruned and pruned-quantized YOLOv5 models. 
In this tutorial, you created a pre-trained model to establish a baseline, applied a Neural Magic recipe for sparsification, and exported to ONNX to run through an inference engine.

Now, refer [here](https://github.com/neuralmagic/deepsparse/tree/main/examples/ultralytics-yolo) for an example for benchmarking and deploying YOLOv5 models with DeepSparse.

For Neural Magic Support, sign up or log in to get help with your questions in our **Tutorials channel**: [Discourse Forum](https://discuss.neuralmagic.com/) and/or [Slack](https://join.slack.com/t/discuss-neuralmagic/shared_invite/zt-q1a1cnvo-YBoICSIw3L1dmQpjBeDurQ). 
