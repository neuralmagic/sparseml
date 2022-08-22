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
and instructions needed to create highly accurate pruned and pruned-quantized [YOLACT](https://arxiv.org/abs/1904.02689) segmentation models.

## Overview

Neural Magic's ML team creates recipes that allow anyone to plug in their data and leverage SparseML's recipe-driven 
approach on top of [YOLACT](https://github.com/dbolya/yolact) training pipelines.
Sparsifying involves removing redundant information from neural networks using algorithms such as pruning and 
quantization, among others. This sparsification process results in many benefits for deployment environments, 
including faster inference and smaller file sizes. Unfortunately, many have not realized the benefits due to the 
complicated process and number of hyper-parameters involved.

Working through this tutorial, you will experience how Neural Magic recipes simplify the sparsification process by:

* Creating a pre-trained model to establish a baseline. You will set up your data and then train the model.
* Applying a recipe to select the trade-off between the amount of recovery to the baseline training performance with 
the amount of sparsification for inference performance.
* Exporting for inference to run a file (that contains a checkpoint of the best weights measured on the validation set) 
through a compression algorithm to reduce its deployment size and run it in an inference engine such as 
[DeepSparse](https://github.com/neuralmagic/deepsparse).

The examples listed in this tutorial are all performed on the COCO dataset.

Before diving in, be sure to go through the setup as listed out in the [README](../README.md) for this integration.

## Need Help?

For Neural Magic Support, sign up or log in to our [**Deep Sparse Community Slack**](https://join.slack.com/t/discuss-neuralmagic/shared_invite/zt-q1a1cnvo-YBoICSIw3L1dmQpjBeDurQ). Bugs, feature requests, or additional questions can also be posted to our [GitHub Issue Queue.](https://github.com/neuralmagic/sparseml/issues)

## Creating a Pre-trained Model

Before applying one of the recipes, you must first create the pre-trained model to sparsify further. 
The pre-trained model enables pruning and other algorithms to remove the correct redundant information in place of random information. 
Your goal after this is to create a smaller, faster model that recovers to the pre-trained baseline.
 
Creating a pre-trained model involves three steps: 
1) Setting up the data.
2) Fetching a model backbone. 
3) Training the model.

If training with COCO, skip steps 2 and 3 and use the following baseline SparseZoo stub,
`zoo:cv/segmentation/yolact-darknet53/pytorch/dbolya/coco/base-none` with the `--resume` argument
in the [`train.py` script](https://github.com/neuralmagic/yolact/blob/master/train.py)

**Note**: If using your custom data, the [YOLACT](https://github.com/dbolya/yolact) repo mentions a post for [training custom data](https://github.com/dbolya/yolact/issues/70#issuecomment-504283008). 
Otherwise, setup scripts for [COCO](https://cocodataset.org/#home) can be found under the [yolact/data/scripts path](https://github.com/neuralmagic/yolact/tree/master/data/scripts).

### Setting Up the Data

1. For this tutorial, run the COCO setup script with the following command from the root of the `yolact` repository:
    ```bash
    sparseml.yolact.download
    ```
   
Based on your internet connection, downloading and validation of the COCO dataset will take around 10 minutes to complete.
    The script downloads the COCO dataset into a `coco` folder under the data directory.
Once completed, the data is ready for training with the folder structure in the following state (only directories are shown for brevity):
```
    └─ data
       └─ coco
          ├── annotations
          └── images
 ```
   
 You are ready to train the model.

### Downloading Model Backbone

1. Training YOLACT from scratch requires a pretrained-backbone model, 
   currently SparseML supports training with `DarkNet-53` backbone. 
   - Download ImageNet-pre-trained [`Darknet-53` backbone](https://drive.google.com/file/d/17Y431j4sagFpSReuPNoFcj9h7azDTZFf/view?usp=sharing)
     and put it in `./weights` directory. The directory structure should look like the following:
````
    └── Project_directory
        ├── data
        │   └─ coco
        │      ├── annotations
        │      └── images
        │   
        └── weights
            └── darknet53.pth
````

2) Run the following command to kickstart training
```bash
sparseml.yolact.train --resume \
./weights/darknet53.pth \
--train_info ./data/coco/annotations/instances_train2017.json \
--validation_info ./data/coco/annotations/instances_val2017.json \
--train_images ./data/coco/images \
--validation_images ./data/coco/images \
--backbone
```
The weights are stored in the `./weights` directory by default and use the `<config>_<epoch>_<iter>.pth` naming 
convention

3) Validate that the training command was completed successfully by checking under the `./weights` directory for the trained weights.
   Upon success, the resulting directory structure should look like the following (a few directories are missing content for brevity):
```
└── Project_Directory
    ├── data
    │    └── coco
    │        ├── annotations
    │        └── images
    │   
    └── weights
        ├── darknet53.pth
        └── yolact_darknet53_54_800000.pth
 ```

You are ready to use the weights at `./weights/yolact_darknet53_54_800000.pth` with the Neural Magic recipes to create a sparsified model.
You can also download this baseline, pre-trained checkpoint directly from the [SparseZoo UI](https://sparsezoo.neuralmagic.com/models/cv%2Fsegmentation%2Fyolact-darknet53%2Fpytorch%2Fdbolya%2Fcoco%2Fbase-none),
or pass its model stub directly to the `--resume` argument while invoking the training script.
## Applying a Recipe

In general, recipes trade off the amount of recovery to the baseline training performance with the amount of sparsification for inference performance.
The [`recipes` folder](../recipes) contains multiple files, each offering certain advantages over others. 
The table below compares these tradeoffs and shows how to run them on the COCO dataset.
1. Review this table, which lists recipes, commands, and results.

| Sparsification Type |                                    Description                                    | COCO mAP@all | Size on Disk | DeepSparse Performance** |                                        Commands                                        |
|:-------------------:|:---------------------------------------------------------------------------------:|:------------:|:------------:|:------------------------:|:--------------------------------------------------------------------------------------:|
| Baseline            | The baseline, pretrained model on the COCO dataset.                               | 0.288        | 170 MB       | 29.7 img/sec               | `python train.py`                                                                      |
| Pruned              | A highly sparse, FP32 model that recovers close to the baseline model.            | 0.286        | 30.1 MB      | 61.6 img/sec               | `python train.py --resume weights/model.pth --recipe ../recipe/yolact.pruned.md`       |
| Pruned Quantized    | A highly sparse, INT8 model that recovers reasonably close to the baseline model. | 0.282        | 9.7 MB       | 144.4 img/sec               | `python train.py --resume weights/model.pth --recipe ../recipe/yolact.pruned_quant.md` |

   \*\* DeepSparse Performance measured on an AWS c5.12xlarge instance with 24 cores, batch size 64, and 550x550 input with version 0.12.0 of the DeepSparse Engine i.e. `deepsparse.benchmark --batch_size 64 --scenario sync [model_path]`
   
2. Select a recipe to use on top of the pre-trained model you created.

    - Check your CPU hardware support for quantized networks (VNNI instruction set) using the DeepSparse API:
      ```bash
      deepsparse.check_hardware
      ```
    - If your hardware does not support quantized networks for inference speedup or complete recovery is very important, then Neural Magic recommends using the  `pruned` recipe. The recipe to use depends on how long you are willing to train and how vital full recovery is. Consult the table above for this comparison.
    - If your hardware does support quantized networks, we recommend using the `pruned quantized` recipe. The recipe to use depends on how long you are willing to train and how crucial full recovery is. Consult the table for this comparison.
    - When running quantized models, the memory footprint for training will significantly increase (roughly 3x). It is recommended to train at a high batch size at first. This will fail with an out-of-memory exception once quantization starts. Once this happens, use the weights from that run to resume training with lower batch size.

3. To begin applying one of the recipes, use the `--recipe` argument within the YOLACT [train script](https://github.com/neuralmagic/yolact/blob/master/train.py).
   The recipe argument is combined with our previous training command and COCO pre-trained weights to run the recipes over the model. For example, a command for pruning YOLACT would look like this:
```bash
sparseml.yolact.train \
--recipe=../recipes/yolact.pruned.md \
--resume=zoo:cv/segmentation/yolact-darknet53/pytorch/dbolya/coco/base-none \
--save_folder=./pruned \
--train_info ./data/coco/annotations/instances_train2017.json \
--validation_info ./data/coco/annotations/instances_val2017.json \
--train_images ./data/coco/images \
--validation_images ./data/coco/images
```
After applying a recipe, you are ready to export for inference.

## Exporting for Inference

This step loads a checkpoint file along with the recipe used if any and converts it into the more common inference formats. 
Then, you can run the file through a compression algorithm to reduce its deployment size and run it in an inference engine such as [DeepSparse](https://github.com/neuralmagic/deepsparse).

When you applied a recipe in the previous step, the sparsification run created a new `./pruned` directory under the `yolact` directory:

```
└── Project_directory
    ├── data
    │   ├── coco
    │   │   ├── annotations
    │   │   └── images
    │   └── scripts
    ├── pruned
    │    └── yolact_darknet53_3_29316.pth 
    └── weights
        └── yolact_darknet53_54_800000.pth
```

These weights under `./pruned` can be loaded into the `sparseml.yolact.train` and `sparseml.yolact.export_onnx` utilities now. 
However, other formats are generally more friendly for other inference deployment platforms, such as [ONNX](https://onnx.ai/).

[`sparseml.yolact.export_onnx`](https://github.com/neuralmagic/yolact/blob/master/export.py) handles the logic behind loading the checkpoint and converting it into the more common inference formats, as described here.

1. Enter the following command to load the PyTorch graph, convert to ONNX, and correct any misformatted pieces of the graph for the pruned and quantized models.

    ```bash
    sparseml.yolact.export_onnx --checkpoint PATH_TO_SPARSIFIED_WEIGHTS
    ```
    
    The result is a new file added next to the sparsified checkpoint with a `.onnx` extension:

2. Now you can run the `.onnx` file through a compression algorithm to reduce its deployment size and run it in ONNX-compatible inference engines such as [DeepSparse](https://github.com/neuralmagic/deepsparse).
   The DeepSparse Engine is explicitly coded to support running sparsified models for significant improvements in inference performance. 
   An example for benchmarking and deploying YOLACT models with DeepSparse can be found [here](https://github.com/neuralmagic/deepsparse/tree/main/examples/dbolya-yolact).

## Wrap-Up

Neural Magic recipes simplify the sparsification process by encoding the hyperparameters and instructions needed to create highly accurate pruned and pruned-quantized YOLACT models for image segmentation tasks. 
In this tutorial, you created a pre-trained model to establish a baseline, applied a Neural Magic recipe for sparsification, and exported it to ONNX to run through an inference engine.

Now, refer [here](https://github.com/neuralmagic/deepsparse/blob/main/src/deepsparse/yolact/README.md) for an example for benchmarking and deploying YOLACT models with DeepSparse.

For Neural Magic Support, sign up or log in to our [**Deep Sparse Community Slack**](https://join.slack.com/t/discuss-neuralmagic/shared_invite/zt-q1a1cnvo-YBoICSIw3L1dmQpjBeDurQ). Bugs, feature requests, or additional questions can also be posted to our [GitHub Issue Queue.](https://github.com/neuralmagic/sparseml/issues)

## Citation

```bibtex
@inproceedings{yolact-iccv2019,
  author    = {Daniel Bolya and Chong Zhou and Fanyi Xiao and Yong Jae Lee},
  title     = {YOLACT: {Real-time} Instance Segmentation},
  booktitle = {ICCV},
  year      = {2019},
}
```
