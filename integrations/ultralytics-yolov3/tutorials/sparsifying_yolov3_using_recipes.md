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

Sparsifying involves removing redundant information from neural networks using algorithms such as pruning and quantization, among others. 
This sparsification process results in many benefits for deployment environments, including faster inference and smaller file sizes.
Unfortunately, many have not realized the benefits due to the complicated process and number of hyperparameters involved. 

Neural Magic's ML team created recipes encoding the necessary hyperparameters and instructions to create highly accurate pruned and pruned-quantized YOLOv3 models to simplify the process.
These recipes allow anyone to plug in their data and leverage SparseML's recipe-driven approach on top of Ultralytics' robust training pipelines.

The examples listed in this tutorial are all performed on the VOC dataset.
Additionally, the results listed in this tutorial are available publicly through a [Weights and Biases project](https://wandb.ai/neuralmagic/yolov3-spp-lrelu-voc).

<div style="width: 100%; display: flex; justify-content: center;">
    <a href="https://youtu.be/o5qIYs47MPw" target="_blank">
        <img alt="Example YOLOv3 Inference Video" src="https://raw.githubusercontent.com/neuralmagic/sparseml/main/integrations/ultralytics-yolov3/tutorials/images/pruned-quantized-result.jpeg" width="560px" style="border: 2px solid #000000;" />
    </a>
</div>

## Creating a Pretrained Model

Before applying one of the recipes, we must first create the pre-trained model to sparsify further.
The pre-trained model enables pruning and other algorithms to remove the correct redundant information in place of random information.
Our goal after this is to create a smaller, faster model that recovers to our pre-trained baseline.

### Setting Up the Data

If using your custom data, the Ultralytics repo contains a walk-through for [training custom data](https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data).
Otherwise, setup scripts for both [VOC](https://cs.stanford.edu/~roozbeh/pascal-context/) and [COCO](https://cocodataset.org/#home) can be found under the [yolov3/data/scripts path](https://github.com/ultralytics/yolov3/tree/master/data/scripts).

For this tutorial, we run the VOC setup script with the following command from the root of the yolov3 repository:
```bash
bash data/scripts/get_voc.sh
```
Download and validation of the VOC dataset will begin and takes around 10 minutes to finish.
The script downloads the VOC dataset into a `VOC` folder under the parent directory.
Once completed, the data is ready for training with the folder structure in the following state:
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

### Training the Model

To expedite the training process, we transfer learn from weights initially trained on the COCO dataset.
These are stored in the SparseZoo and can be accessed with the following Python code:
```python
from sparsezoo import Zoo

model = Zoo.load_model_from_stub("zoo:cv/detection/yolo_v3-spp/pytorch/ultralytics/coco/base-none")
checkpoint_path = model.framework_files[0].downloaded_path()
print(checkpoint_path)
```

After running, the code has downloaded the checkpoint file to the local system:
```bash
downloading...: 100%|██████████| 126342273/126342273 [00:04<00:00, 28306365.97it/s]
/Users/markkurtz/.cache/sparsezoo/16fe8358-2e91-4b81-a1f2-df85bd1a9ac3/pytorch/model.pth
```

Next, the checkpoint file provides the source weights for training on VOC using the following train command run from within the yolov3 repository folder:
```bash
python train.py --weights PATH_TO_COCO_PRETRAINED_WEIGHTS --data voc.yaml --img 512 --epochs 50
```

The training command will take a few hours to complete (~3 hours for a titan RTX).
Afterward, we'll have a model that achieves roughly 0.85 mAP on the VOC dataset ready for sparsifying.

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

Now we are ready to use the weights at `yolov3/runs/train/exp/weights/best.pt` with our recipes to create sparsified models!

## Applying a Recipe

In general, recipes trade off the amount of recovery to the baseline training performance with the amount of sparsification for inference performance.
The [`recipes` folder](https://github.com/neuralmagic/sparseml/blob/main/integrations/ultralytics-yolov3/recipes) contains multiple files, each offering certain advantages over others.
Below we walk through each to compare these tradeoffs and show how to run them on the VOC dataset.

To begin applying one of the recipes, we use the `--recipe` argument within the Ultralytics [train script](https://github.com/neuralmagic/yolov3/blob/master/train.py).
In addition, the hyperparameters are changed slightly to better work with the recipe.
These hyperparameters are stored in appropriately named files under the [`data` directory](https://github.com/neuralmagic/sparseml/blob/main/integrations/ultralytics-yolov3/data) and are passed into the training script using the `--hyp` argument.
Both of the arguments are combined with our previous training command and VOC pre-trained weights to run the recipes over the model.
The recipes, commands, and results are all listed in the table below.

| Recipe Name                                                                                                                                              | Description                                                                                                                     | Train Command                                                                                                                                                                               | VOC mAP@0.5 | Size on Disk | DeepSparse Performance** | Training Epochs (time) | Weights and Biases                                                       |
|----------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------|--------------|--------------------------|------------------------|--------------------------------------------------------------------------|
| Baseline                                                                                                                                                 | The baseline, pretrained model originally transfer learned onto the VOC dataset.                                                | ``` python train.py --weights PATH_TO_COCO_PRETRAINED_WEIGHTS --data voc.yaml --img 512 --epochs 50 ```                                                                                     | 0.857       | 194 MB       | 19.5 img/sec             | 50 (3.21 hours)        | [wandb](https://wandb.ai/neuralmagic/yolov3-spp-lrelu-voc/runs/261llnb1) |
| [Pruned](https://github.com/neuralmagic/sparseml/blob/main/integrations/ultralytics-yolov3/recipes/yolov3-spp.pruned.md)                                 | Creates a highly sparse, FP32 model that recovers close to the baseline model.                                                  | ``` python train.py --weights PATH_TO_VOC_PRETRAINED_WEIGHTS --data voc.yaml --img 512 --hyp ../data/hyp.pruned.yaml --recipe ../recipes/yolov3-spp.pruned.md ```                           | 0.858       | 33.6 MB      | 34.0 img/sec             | 300 (20.45 hours)      | [wandb](https://wandb.ai/neuralmagic/yolov3-spp-lrelu-voc/runs/2jeadrts) |
| [Pruned Short](https://github.com/neuralmagic/sparseml/blob/main/integrations/ultralytics-yolov3/recipes/yolov3-spp.pruned.short.md)                     | Creates a highly sparse, FP32 model in a shortened schedule to prioritize quicker training while sacrificing a bit on recovery. | ``` python train.py --weights PATH_TO_VOC_PRETRAINED_WEIGHTS --data voc.yaml --img 512 --hyp ../data/hyp.pruned.yaml --recipe ../recipes/yolov3-spp.pruned.short.md ```                     | 0.854       | 33.6 MB      | 34.0 img/sec             | 80 (5.53 hours)        | [wandb](https://wandb.ai/neuralmagic/yolov3-spp-lrelu-voc/runs/jktw650n) |
| [Pruned Quantized](https://github.com/neuralmagic/sparseml/blob/main/integrations/ultralytics-yolov3/recipes/yolov3-spp.pruned_quantized.md)             | Creates a highly sparse, INT8 model that recovers reasonably close to the baseline model.                                       | ``` python train.py --weights PATH_TO_VOC_PRETRAINED_WEIGHTS --data voc.yaml --img 512 --hyp ../data/hyp.pruned_quantized.yaml --recipe ../recipes/yolov3-spp.pruned_quantized.md ```       | 0.827       | 13.4 MB      | 90.4 img/sec             | 242 (18.32 hours)      | [wandb](https://wandb.ai/neuralmagic/yolov3-spp-lrelu-voc/runs/2dfy3rgs) |
| [Pruned Quantized Short](https://github.com/neuralmagic/sparseml/blob/main/integrations/ultralytics-yolov3/recipes/yolov3-spp.pruned_quantized.short.md) | Creates a highly sparse, INT8 model in a shortened schedule to prioritize quicker training while sacrificing a bit on recovery. | ``` python train.py --weights PATH_TO_VOC_PRETRAINED_WEIGHTS --data voc.yaml --img 512 --hyp ../data/hyp.pruned_quantized.yaml --recipe ../recipes/yolov3-spp.pruned_quantized.short.md ``` | 0.808       | 13.4 MB      | 90.4 img/sec             | 52 (4.23 hours)        | [wandb](https://wandb.ai/neuralmagic/yolov3-spp-lrelu-voc/runs/1picfimy) |
| [Test](https://github.com/neuralmagic/sparseml/blob/main/integrations/ultralytics-yolov3/recipes/yolov3-spp.test.md)                                     | A test recipe to test the training pipeline and device for both pruning and quantization in 5 epochs.                           | ``` python train.py --weights PATH_TO_VOC_PRETRAINED_WEIGHTS --data voc.yaml --img 512 --hyp ../data/hyp.pruned_quantized.yaml --recipe ../recipes/yolov3-spp.test.md ```                   | 0.702       | 13.4 MB      | 90.4 img/sec             | 5 (17 minutes)         | [wandb](https://wandb.ai/neuralmagic/yolov3-spp-lrelu-voc/runs/3bkw6c60) |

** DeepSparse Performance measured on an aws C5 instance with 24 cores, batch size 64, and 640x640 input with version 1.3 of the DeepSparse Engine.

Weights and Biases is very useful for comparing across these runs, the epochs vs mAP@0.5 graph is supplied below:

<img src="https://raw.githubusercontent.com/neuralmagic/sparseml/main/integrations/ultralytics-yolov3/tutorials/images/pruned-quantized-wandb-chart.png" width="960px" style="border: 2px solid #000000;" />

If your hardware does not support quantized networks for inference speedup or complete recovery is very important, then we recommend using either the `pruned` or `pruned short` recipe.
Which one to use depends on how long you are willing to train and how vital full recovery is.
Consult the table above for this comparison.

If your hardware does support quantized networks (VNNI instruction set on CPUs, for example), we recommend using the `pruned quantized` or `pruned quantized short` recipe.
Which one to use depends on how long you are willing to train and how crucial full recovery is.
Consult the table above for this comparison.

When running quantized models, the memory footprint for training will significantly increase (roughly 3x).
Therefore it is essential to take this into account when selecting a batch size to train at.
To ensure no issues with the longer quantized runs, please run the quicker test recipe first to ensure your configurations are correct and the training process will complete successfully.

## Exporting for Inference

The sparsification run will create a new `exp` directory under the yolov3 runs directory:
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
These weights can be loaded into the `train.py` and `test.py` scripts now.
However, other formats are generally more friendly for other inference deployment platforms, such as [ONNX](https://onnx.ai/).

The [`export.py` script](https://github.com/neuralmagic/yolov3/blob/master/models/export.py) handles the logic behind loading the checkpoint and converting it into the more common inference formats.
The following command loads the PyTorch graph, converts to ONNX, and then corrects any misformatted pieces for the pruned and quantized models.
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

Now you can run the `.onnx` file through a compression algorithm to reduce its deployment size and run it in ONNX-compatible inference engines such as [DeepSparse](https://github.com/neuralmagic/deepsparse).
The DeepSparse Engine is explicitly coded to support running sparsified models for significant improvements in inference performance.
An example for benchmarking and deploying YOLOv3 models with DeepSparse can be found [here](https://github.com/neuralmagic/deepsparse/tree/main/examples/ultralytics-yolov3).
