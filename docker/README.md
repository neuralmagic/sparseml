# SparseML Docker Image
This directory contains the Dockerfile to create a minimal SparseML docker image.

The included `Dockerfile` builds an image on top of the official NVIDIA development Ubuntu 18.04.5 LTS 
image (with CUDA support). The image comes with a pre-installed `sparseml`, as well as the `torch==1.9.1`
(compatible with either CUDA 10.2 or CUDA 11.1).

This `Dockerfile` is tested on the Ubuntu 20.04.2 LTS with CUDA Version: 11.4.

## Pull
You can access the already built image detailed at https://github.com/orgs/neuralmagic/packages/container/package/sparseml:

```bash
docker pull ghcr.io/neuralmagic/sparseml:1.4.4-cu111
docker tag ghcr.io/neuralmagic/sparseml:1.4.4-cu111 sparseml_docker
```

## Extend
If you would like to customize the docker image, you can use the pre-built images as a base in your own `Dockerfile`:

```Dockerfile
from ghcr.io/neuralmagic/sparseml:1.4.4-cu111

...
```

## Build
To build and launch this image with the tag `sparseml_docker`, run from the root directory: `docker build -t sparseml_docker`

If you want to use a specific branch from sparseml you can use the `BRANCH` build arg:
```bash
docker build --build-arg BRANCH=main -t sparseml_docker .
```

## Run
To run the built image launch: `docker run -it --gpus all sparseml_docker`
This will run the bash console in the container's `root` directory.

## Examples
Once inside the container, the `sparseml` can be used right away.

Note: RuntimeError: DataLoader worker (pid 1388) is killed by signal: Bus error. It is possible that dataloader's workers are out of shared memory. Please try to raise your shared memory limit.

### Example 1: Image Classification Pipeline:

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

### Example 2: Transformers Question Answering Pipeline:

```bash
sparseml.transformers.question_answering \
  --model_name_or_path bert-base-uncased \
  --dataset_name squad \
  --do_train \
  --do_eval \
  --output_dir './output' \
  --cache_dir cache \
  --distill_teacher disable \
  --recipe zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned-aggressive_98 
```

Note: 
If any of the examples results in an error:
```
RuntimeError: DataLoader worker {...} is killed by signal: Bus error. It is possible that dataloader's workers are out of shared memory. Please try to raise your shared memory limit.`
```
either:
- use a smaller batch size to train your model.
- re-run the docker with the `--shm-size` argument specified (to increase the shared memory space):
```
docker run -it --gpus all --shm-size 16g sparseml_docker
```
