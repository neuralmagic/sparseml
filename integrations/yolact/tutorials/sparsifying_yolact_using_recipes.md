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

# Sparsifying YOLACT Using Recipes

This tutorial shows how Neural Magic recipes simplify the sparsification process by encoding the hyperparameters 
and instructions needed to create highly accurate pruned and pruned-quantized YOLACT segmentation models.

## Overview

Neural Magic's ML team creates recipes that allow anyone to plug in their data and leverage SparseML's recipe-driven 
approach on top of [YOLACT](https://github.com/dbolya/yolact) training pipelines.
Sparsifying involves removing redundant information from neural networks using algorithms such as pruning and 
quantization, among others. This sparsification process results in many benefits for deployment environments, 
including faster inference and smaller file sizes. Unfortunately, many have not realized the benefits due to the 
complicated process and number of hyperparameters involved.

Working through this tutorial, you will experience how Neural Magic recipes simplify the sparsification process by:

* Creating a pre-trained model to establish a baseline. You will set up your custom data and then train the model.
* Applying a recipe to select the trade off between the amount of recovery to the baseline training performance with 
the amount of sparsification for inference performance.
* Exporting for inference to run a file (that contains a checkpoint of the best weights measured on the validation set) 
through a compression algorithm to reduce its deployment size and run it in an inference engine such as 
[DeepSparse](https://github.com/neuralmagic/deepsparse).

The examples listed in this tutorial are all performed on the COCO dataset.

Before diving in, be sure to go through setup as listed out in the [README](../README.md) for this integration.
Additionally, all commands are intended to be run from the root of the `yolact` repository folder 
(`cd integrations/yolact/yolact`).

## Need Help?

For Neural Magic Support, sign up or log in to get help with your questions in our **Tutorials channel**: [Discourse Forum](https://discuss.neuralmagic.com/) and/or [Slack](https://join.slack.com/t/discuss-neuralmagic/shared_invite/zt-q1a1cnvo-YBoICSIw3L1dmQpjBeDurQ). 

## Creating a Pre-trained Model

Before applying one of the recipes, you must first create the pre-trained model to sparsify further. 
The pre-trained model enables pruning and other algorithms to remove the correct redundant information in place of random information. 
Your goal after this is to create a smaller, faster model that recovers to the pre-trained baseline.

Creating a pre-trained model involves two steps: 1) setting up the data and 2) training the model.

**Note**: If using your custom data, the [YOLACT](https://github.com/dbolya/yolact) repo mentions a post for [training custom data](https://github.com/dbolya/yolact/issues/70#issuecomment-504283008). 
Otherwise, setup scripts for [COCO](https://cocodataset.org/#home) can be found under the [yolact/data/scripts path](https://github.com/dbolya/yolact/tree/main/data/scripts).

### Setting Up the Data

1. For this tutorial, run the COCO setup script with the following command from the root of the `yolact` repository:
    ```bash
    bash data/scripts/COCO.sh
    ```
2. Download and validation of the COCO dataset will begin and takes around 10 minutes to finish(Based on the speed of your Internet Connection).
    The script downloads the COCO dataset into a `coco` folder under the data directory.
    Notice that, once completed, the data is ready for training with the folder structure in the following state(Only directories shown for brevity):
    ```
   └── yolact
    ├── data
    │   ├── coco
    │   │   ├── annotations
    │   │   └── images
    │   └── scripts
    ├── external
    │   └── DCNv2
    │       └── src
    ├── layers
    │   ├── functions
    │   └── modules
    ├── scripts
    ├── utils
    └── web
        ├── css
        ├── dets
        └── scripts
    ```
   
    You are ready to train the model.

### Training the Model

The training command will take multiple hours to complete since it is training from scratch. 
Afterward, you will have a model that achieves roughly  on the COCO dataset ready for sparsifying.

(You can also download a pretrained model with a darkent53 backbone from the YOLACT repository, in that case skip step 1)

1. The training command for training from scratch, with a default batch_size of 8:
   
   ```bash
   python train.py --config yolact_darknet53_config 
   ```    
    The weights are stored in the `./weights` directory by default and use `<config>_<epoch>_<iter>.pth` naming 
    convention

2. Validate that the training commands completed successfully by checking under the `./weights` directory for the trained weights.
   Upon success, the results directory should look like the following (A few directories are missing content for brevity):
```
└── yolact
    ├── data
    │   ├── coco
    │   │   ├── annotations
    │   │   └── images
    │   └── scripts
    ├── external
    │   └── DCNv2
    │       └── src
    ├── layers
    │   ├── functions
    │   └── modules
    ├── scripts
    ├── utils
    ├── web
    │   ├── css
    │   ├── dets
    │   └── scripts
    └── weights
        └── yolact_darknet53_54_800000.pth
 ```

You are ready to use the weights at `yolact/weights/yolact_darknet53_54_800000.pth` with the Neural Magic recipes to create a sparsified model.

## Applying a Recipe

In general, recipes trade off the amount of recovery to the baseline training performance with the amount of sparsification for inference performance.
The [`recipes` folder](../recipes) contains multiple files, each offering certain advantages over others. 
The table below compares these tradeoffs and shows how to run them on the COCO dataset.
1. Review this table, which lists recipes, commands, and results.

    | Recipe Name                                                                                                                                              | Description                                                                                                                     | Train Command                                                                                                                                                                               | COCO mAP@0.5 box/mask | Size on Disk | DeepSparse Performance** |
    |----------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------|--------------|--------------------------|
    | YOLACT Baseline                                                                                                                                         | The baseline, YOLACT image segmentation model used as the starting point for sparsification.                                     | ```python train.py --config yolact_darknet53_config```                                                                              |  30.61/28.80 |              |             |
    
    ** DeepSparse Performance measured on an AWS C5 instance with 24 cores, batch size 64, and 550 x 550 input with version 1.6 of the DeepSparse Engine.

2. Select a recipe to use on top of the pre-trained model you created.

    - Check your CPU hardware support for quantized networks (VNNI instruction set) using the DeepSparse API:
      ```python
      from deepsparse.cpu import cpu_vnni_compatible
      print(f"VNNI available: {cpu_vnni_compatible()}")
      ```
    - If your hardware does not support quantized networks for inference speedup or complete recovery is very important, then Neural Magic recommends using the  `pruned` recipe. The recipe to use depends on how long you are willing to train and how vital full recovery is. Consult the table above for this comparison.
    - If your hardware does support quantized networks, we recommend using the `pruned quantized` recipe. The recipe to use depends on how long you are willing to train and how crucial full recovery is. Consult the table for this comparison.
    - When running quantized models, the memory footprint for training will significantly increase (roughly 3x). It is recommended to train at a high batch size at first. This will fail with an out-of-memory exception once quantization starts. Once this happens, use the `last.pt` weights from that run to resume training with a lower batch size.

3. To begin applying one of the recipes, use the `--recipe` argument within the YOLACT [train script](https://github.com/neuralmagic/yolact/blob/main/train.py).
   The recipe argument is combined with our previous training command and COCO pre-trained weights to run the recipes over the model. For example, a command for YOLACT would look like this:
```bash
python train.py \
--config=yolact_darknet53_config \
--recipe=./recipes/yolact.quant.yaml \
--resume=./weights/yolact_darknet53_54_800000.pth \
--cuda=True \
--start_iter=0 \
--save_folder=./dense-quantized \
--batch_size=8 
```
After applying a recipe, you are ready to export for inference.

## Exporting for Inference

This step loads a checkpoint file along with the recipe used if any, and converts it into the more common inference formats. 
Then, you can run the file through a compression algorithm to reduce its deployment size and run it in an inference engine such as [DeepSparse](https://github.com/neuralmagic/deepsparse).

When you applied a recipe in the previous step, the sparsification run created a new `./dense-quantized` directory under the `yolact` directory:

```
└── yolact
    ├── data
    │   ├── coco
    │   │   ├── annotations
    │   │   └── images
    │   └── scripts
    ├── dense-quantized
    │    └── yolact_darknet53_3_29316.pth 
    ├── external
    │   └── DCNv2
    │       └── src
    ├── layers
    │   ├── functions
    │   └── modules
    ├── scripts
    ├── utils
    ├── web
    │   ├── css
    │   ├── dets
    │   └── scripts
    └── weights
        └── yolact_darknet53_54_800000.pth
```

These weights under `./dense-quantized` can be loaded into the `train.py` and `export.py` scripts now. 
However, other formats are generally more friendly for other inference deployment platforms, such as [ONNX](https://onnx.ai/).

The [`export.py` script](https://github.com/neuralmagic/yolact/blob/main/models/export.py) handles the logic behind loading the checkpoint and converting it into the more common inference formats, as described here.

1. Enter the following command to load the PyTorch graph, convert to ONNX, and correct any misformatted pieces of the graph for the pruned and quantized models.

    ```bash
    python export.py --weights PATH_TO_SPARSIFIED_WEIGHTS --recipe PATH_TO_RECIPE_IF_USED
    ```
    
    The result is a new file added next to the sparsified checkpoint with a `.onnx` extension:

2. Now you can run the `.onnx` file through a compression algorithm to reduce its deployment size and run it in ONNX-compatible inference engines such as [DeepSparse](https://github.com/neuralmagic/deepsparse).
   The DeepSparse Engine is explicitly coded to support running sparsified models for significant improvements in inference performance. 
   An example for benchmarking and deploying YOLACT models with DeepSparse can be found [here](https://github.com/neuralmagic/deepsparse/tree/main/examples/yolact).

## Wrap-Up

Neural Magic recipes simplify the sparsification process by encoding the hyperparameters and instructions needed to create highly accurate pruned and pruned-quantized YOLACT models for image segmentation tasks. 
In this tutorial, you created a pre-trained model to establish a baseline, applied a Neural Magic recipe for sparsification, and exported to ONNX to run through an inference engine.

Now, refer [here](https://github.com/neuralmagic/deepsparse/tree/main/examples/yolact) for an example for benchmarking and deploying YOLACT models with DeepSparse.

For Neural Magic Support, sign up or log in to get help with your questions in our **Tutorials channel**: [Discourse Forum](https://discuss.neuralmagic.com/) and/or [Slack](https://join.slack.com/t/discuss-neuralmagic/shared_invite/zt-q1a1cnvo-YBoICSIw3L1dmQpjBeDurQ). 
