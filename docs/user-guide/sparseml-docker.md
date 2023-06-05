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

# How to Use SparseML With Docker 
SparseML provides libraries for applying sparsification recipes to neural networks with a few lines of code, enabling faster and smaller models. 

Apart from installing SparseML via `pip` you can set it up quickly using Docker. 

## Prerequisites
Before you begin, make sure you have Docker installed on your machine. You can download and install it from the [official Docker website](https://www.docker.com/products/docker-desktop).

## Pull The Official SparseML Image

The following lines of code will: 
- Pull the official SparseML image from GitHub Container Registry 
- Tag the image as `sparseml_docker` 
- Start the `sparseml_docker` in interactive mode

```bash
docker pull ghcr.io/neuralmagic/sparseml:1.4.4-cu111
docker tag ghcr.io/neuralmagic/sparseml:1.4.4-cu111 sparseml_docker
docker container run -it sparseml_docker
```
## NLP NER Example
You can train various CV or NLP models inside the SparseML container. To use GPUs when training, add the `gpus` flag when starting the container.

The command below starts the container with all the available GPUs: 
```bash 
docker container run -it --gpus all sparseml_docker
```
You can also start the container with a higher memory allocation declared with the `shm-size` argument to prevent memory problems:
```bash 
docker container run --gpus all --shm-size=256m -it sparseml_docker
```

Here's an example showing how to train a NER model inside the SparseML container:

```bash
sparseml.transformers.train.token_classification \
  --model_name_or_path zoo:nlp/masked_language_modeling/obert-base/pytorch/huggingface/wikipedia_bookcorpus/pruned90-none \
  --recipe zoo:nlp/token_classification/obert-base/pytorch/huggingface/conll2003/pruned90_quant-none \
  --distill_teacher zoo:nlp/token_classification/obert-base/pytorch/huggingface/conll2003/base-none \
  --dataset_name conll2003 \
  --output_dir sparse_bert-token_classification_conll2003 \
  --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --preprocessing_num_workers 6 \
  --do_train --do_eval --evaluation_strategy epoch --fp16 --seed 29204  \
  --save_strategy epoch --save_total_limit 1

   > ***** eval metrics *****
   > epoch                   =       13.0
   > eval_accuracy           =       0.98
   > eval_f1                 =     0.8953
   > eval_loss               =     0.0878
   > eval_precision          =     0.8887
   > eval_recall             =     0.9021
   > eval_runtime            = 0:00:12.56
   > eval_samples            =       3251
   > eval_samples_per_second =    258.667
   > eval_steps_per_second   =      2.069

```

To confirm that the GPUs are being utilized, run: 

```bash
$ nvidia-smi
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 520.61.05    Driver Version: 520.61.05    CUDA Version: 11.8     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA RTX A4000    On   | 00000000:05:00.0 Off |                  Off |
| 44%   65C    P2   106W / 140W |   9614MiB / 16376MiB |     97%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   1  NVIDIA RTX A4000    On   | 00000000:06:00.0 Off |                  Off |
| 41%   56C    P2    77W / 140W |   6322MiB / 16376MiB |     42%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   2  NVIDIA RTX A4000    On   | 00000000:07:00.0 Off |                  Off |
| 41%   55C    P2    72W / 140W |   6322MiB / 16376MiB |     37%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   3  NVIDIA RTX A4000    On   | 00000000:08:00.0 Off |                  Off |
| 41%   54C    P2    74W / 140W |   6322MiB / 16376MiB |     51%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
```
Lower the `batch_size` in case you get any CUDA error messages.

## Image Classification

The `sparseml.image_classification.train` command is used to train image classification models with SparseML. 

Start the SparseML container with all the available GPUs: 

```bash 
docker container run --gpus all -it sparseml_docker
```

Download a subset of the ImageNet dataset and use it to train a ResNet-50 model. 
```bash 
curl https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz --output imagenette2-320.tgz
tar -xvf imagenette2-320.tgz

sparseml.image_classification.train \
    --recipe zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned95_quant-none?recipe_type=transfer-classification \
    --checkpoint-path zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned95_quant-none?recipe_type=transfer-classification \
    --arch-key resnet50 \
    --dataset-path ./imagenette2-320 \
    --batch-size 32
 ```
 The [`recipe`](https://sparsezoo.neuralmagic.com/models/cv%2Fclassification%2Fresnet_v1-50%2Fpytorch%2Fsparseml%2Fimagenet%2Fpruned95_quant-none) 
 instructs SparseML to maintain sparsity during training and to quantize the model over the final epochs.

## Object Detection
The `sparseml.yolov5.train` command is used to train YOLOv5 models with SparseML. 

Start the SparseML container with all the available GPUs: 

```bash 
docker container run --gpus all -it sparseml_docker
```
The CLI command below trains a YOLOv5 model on the VOC dataset with a [`recipe`](https://sparsezoo.neuralmagic.com/models/cv%2Fdetection%2Fyolov5-s%2Fpytorch%2Fultralytics%2Fcoco%2Fpruned75_quant-none) 
that instructs SparseML to maintain sparsity during training and to quantize the model over the final epochs.
```bash
sparseml.yolov5.train \
  --data VOC.yaml \
  --cfg models_v5.0/yolov5s.yaml \
  --weights zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned_quant-aggressive_94?recipe_type=transfer \
  --hyp data/hyps/hyp.finetune.yaml \
  --recipe zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned-aggressive_96

```
## How to Build Your Own SparseML Image
To build your own SparseML image [follow these instructions](https://github.com/neuralmagic/sparseml/blob/main/docker/README.md)
