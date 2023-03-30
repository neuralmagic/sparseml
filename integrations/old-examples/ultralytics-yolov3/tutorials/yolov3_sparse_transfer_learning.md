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

# Sparse Transfer Learning With YOLOv3

This tutorial shows how Neural Magic sparse models simplify the sparsification process by offering pre-sparsified models for transfer learning onto other datasets.

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
Additionally, the results listed in this tutorial are available publicly through a [Weights and Biases project](https://wandb.ai/neuralmagic/yolov3-spp-voc-sparse-transfer-learning).

<div style="width: 100%; display: flex; justify-content: center;">
    <a href="https://youtu.be/o5qIYs47MPw" target="_blank">
        <img alt="Example YOLOv3 Inference Video" src="https://raw.githubusercontent.com/neuralmagic/sparseml/main/integrations/ultralytics-yolov3/tutorials/images/pruned-quantized-result.jpeg" width="560px" style="border: 2px solid #000000;" />
    </a>
</div>

## Need Help?

For Neural Magic Support, sign up or log in to our [**Deep Sparse Community Slack**](https://join.slack.com/t/discuss-neuralmagic/shared_invite/zt-q1a1cnvo-YBoICSIw3L1dmQpjBeDurQ). Bugs, feature requests, or additional questions can also be posted to our [GitHub Issue Queue.](https://github.com/neuralmagic/sparseml/issues)

## Downloading and Preparing a Pre-pruned Model

First, you need to download the sparsified models from the [SparseZoo](https://github.com/neuralmagic/sparsezoo).
These models were originally pruned on the COCO dataset achieving the following metrics:

| Sparsification Type | Description                                                                       | COCO mAP@0.5 | Size on Disk | DeepSparse Performance** |
|---------------------|-----------------------------------------------------------------------------------|--------------|--------------|--------------------------|
| Baseline            | The baseline, pretrained model on the COCO dataset.                               | 0.642        | 194 MB       | 19.5 img/sec             |
| Pruned              | A highly sparse, FP32 model that recovers close to the baseline model.            | 0.624        | 33.6 MB      | 34.0 img/sec             |
| Pruned Quantized    | A highly sparse, INT8 model that recovers reasonably close to the baseline model. | 0.605        | 13.4 MB      | 90.4 img/sec             |
** DeepSparse Performance measured on an AWS C5 instance with 24 cores, batch size 64, and 640x640 input with version 1.3 of the DeepSparse Engine.

1) After deciding on which model meets your performance requirements for both speed and accuracy, the following code is used to download the PyTorch checkpoints for the desired model from the SparseZoo:
    ```python
    from sparsezoo import Model
    
    BASELINE_STUB = 'zoo:cv/detection/yolo_v3-spp/pytorch/ultralytics/coco/base-none'
    PRUNED_STUB = 'zoo:cv/detection/yolo_v3-spp/pytorch/ultralytics/coco/pruned-aggressive_97'
    PRUNED_QUANT_STUB = 'zoo:cv/detection/yolo_v3-spp/pytorch/ultralytics/coco/pruned_quant-aggressive_94'
    
    stub = PRUNED_QUANT_STUB
    model = Model(stub)
    downloded_path = model.path
    print(f'Model with stub {stub} downloaded to {downloded_path}.')
    ```

2) Once the desired checkpoint has downloaded, it must be reset for training again.
    The `utility.py` script within YOLOv3 repository is used for this.
    First change into the `yolov3` directory using `cd yolov3`.
    Next run the following command to prepare the checkpoint for training:
    ```bash
    python utility.py strip /PATH/TO/DOWNLOADED/WEIGHTS.pt
    ```
   
You are now ready to set up the data for training.

## Setting Up the Data

Note: If using your custom data, the Ultralytics repo contains a walk-through for [training custom data](https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data). 
Otherwise, setup scripts for both [VOC](https://cs.stanford.edu/~roozbeh/pascal-context/) and [COCO](https://cocodataset.org/#home) can be found under the [yolov3/data/scripts path](https://github.com/ultralytics/yolov3/tree/master/data/scripts).

1) For this tutorial, run the VOC setup script with the following command from the root of the `yolov3` repository:
    
    `bash data/scripts/get_voc.sh`
2) Download and validation of the VOC dataset will begin and take around 10 minutes to finish. 
    The script downloads the VOC dataset into a VOC folder under the parent directory. 
    Notice that, once completed, the data is ready for training with the folder structure in the following state:
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

You are ready to transfer learn the model.

## Transfer Learning the Model

The training command will take a few hours to complete (~3 hours for a Titan RTX). 
Afterward, you will have a sparse model transfer learned onto the VOC dataset.

The command uses the `--recipe` argument to encode the proper hyperparams such that SparseML will enforce the sparsity for all layers.
Without the proper recipes, the zeroed weights will slowly become dense as the model is trained further.
The recipes are specific to the sparsification type, so the training command will differ based on if you are transfer learning the pruned quantized, pruned, or baseline model.

1) Select the proper command to run based on the sparsification type of the model you downloaded earlier.
   Change directory into the `yolov3` folder to use the `train.py` script.
   - Pruned transfer learning, achieves 0.84 mAP@0.5 on the VOC dataset:
        ```bash
        python train.py --data voc.yaml --cfg ../models/yolov3-spp.lrelu.yaml --weights DOWNLOADED_PATH --hyp data/hyp.finetune.yaml --recipe ../recipes/yolov3-spp.transfer_learn_pruned.md
        ```
   - Pruned-Quantized transfer learning, achieves 0.838 mAP@0.5 on the VOC dataset:
        ```bash
        python train.py --data voc.yaml --cfg ../models/yolov3-spp.lrelu.yaml --weights DOWNLOADED_PATH --hyp data/hyp.finetune.yaml --recipe ../recipes/yolov3-spp.transfer_learn_pruned_quantized.md
        ```
     Note: Running quantization-aware training (QAT) will consume up to three times the memory as FP16 training. 
     It is recommended to use this recipe with a batch size that fits in memory for FP16 as normal at the start, though.
     This will keep the training times the same until the quantization portion.
     Then it will fail with an out of memory exception, but you can use the `last.pt` checkpoint with the `--weights` argument to resume the run with a lower batch size for QAT.
   - Baseline transfer learning, achieves 0.86 mAP@0.5 on the VOC dataset:
        ```bash
        python train.py --data voc.yaml --cfg ../models/yolov3-spp.lrelu.yaml --weights DOWNLOADED_PATH --hyp data/hyp.finetune.yaml
        ```

    Weights and Biases is very useful for comparing across the different runs; the epochs vs mAP@0.5 graph is supplied below:
    
    <img src="https://raw.githubusercontent.com/neuralmagic/sparseml/main/integrations/ultralytics-yolov3/tutorials/images/transfer-learning-wandb-chart.png" width="960px" style="border: 2px solid #000000;" />

Afterward, you will have a sparse model trained on the VOC dataset almost ready for inference.
The training command creates a `runs` directory under the `yolov3` repository directory. 
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

## Exporting for Inference

This step loads a checkpoint file of the best weights measured on the validation set, and converts it into the more common inference formats. 
Then, you can run the file through a compression algorithm to reduce its deployment size and run it in an inference engine such as DeepSparse.

The `best.pt` file contains a checkpoint of the best weights measured on the validation set. 
These weights can be loaded into the `train.py` and `test.py` scripts now. 
However, other formats are generally more friendly for other inference deployment platforms, such as [ONNX](https://onnx.ai/).

The [export.py script](https://github.com/neuralmagic/yolov3/blob/master/models/export.py) handles the logic behind loading the checkpoint and converting it into the more common inference formats, as described here.
1) Enter the following command to load the PyTorch graph, convert to ONNX, and then correct any misformatted pieces for the pruned and quantized models.
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
2) Now you can run the `.onnx` file through a compression algorithm to reduce its deployment size and run it in ONNX-compatible inference engines such as [DeepSparse](https://github.com/neuralmagic/deepsparse).
    The DeepSparse Engine is explicitly coded to support running sparsified models for significant improvements in inference performance. 
    An example for benchmarking and deploying YOLOv3 models with DeepSparse can be found [here](https://github.com/neuralmagic/deepsparse/tree/main/examples/ultralytics-yolo).

## Wrap-Up

Neural Magic sparse models and recipes simplify the sparsification process by enabling sparse transfer learning to create highly accurate pruned and pruned-quantized YOLOv3 models. 
In this tutorial, you downloaded a pre-sparsified model, applied a Neural Magic recipe for sparse transfer learning, and exported to ONNX to run through an inference engine.

Now, refer [here](https://github.com/neuralmagic/deepsparse/tree/main/examples/ultralytics-yolo) for an example for benchmarking and deploying YOLOv3 models with DeepSparse.

For Neural Magic Support, sign up or log in to our [**Deep Sparse Community Slack**](https://join.slack.com/t/discuss-neuralmagic/shared_invite/zt-q1a1cnvo-YBoICSIw3L1dmQpjBeDurQ). Bugs, feature requests, or additional questions can also be posted to our [GitHub Issue Queue.](https://github.com/neuralmagic/sparseml/issues)
