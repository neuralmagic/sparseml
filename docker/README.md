# SparseML docker image
This directory contains the Dockerfile to create a minimal SparseML docker image.

The included `Dockerfile` builds an image on top of the official NVIDIA development Ubuntu 18.04.5 LTS 
image (with CUDA support). The image comes with a pre-installed `sparseml`, as well as the `torch==1.9.1`
(compatible with either CUDA 10.2 or CUDA 11.1).

This `Dockerfile` is tested on the Ubuntu 20.04.2 LTS with CUDA Version: 11.4.

## Pull
You can access the already built image detailed at https://github.com/orgs/neuralmagic/packages/container/package/sparseml:

```bash
docker pull ghcr.io/neuralmagic/sparseml:1.0.1-ubuntu18.04-cu11.1
docker tag ghcr.io/neuralmagic/sparseml:1.0.1-ubuntu18.04-cu11.1 sparseml_docker
```

## Extend
If you would like to customize the docker image, you can use the pre-built images as a base in your own `Dockerfile`:

```Dockerfile
from ghcr.io/neuralmagic/sparseml:1.0.1-ubuntu18.04-cu11.1

...
```

## Build
To build and launch this image with the tag `sparseml_docker`, run from the root directory:
- for compute platform CUDA 10.2: `docker build --build-arg CUDA_VERSION=10.2 -t sparseml_docker .`
- for compute platform CUDA 11.1: `docker build --build-arg CUDA_VERSION=11.1 -t sparseml_docker .` 

If you want to use a specific branch from sparseml you can use the `GIT_CHECKOUT` build arg:
```
docker build --build-arg CUDA_VERSION=11.1 --build-arg GIT_CHECKOUT=main -t sparseml_nightly .`
```

## Run
To run the built image launch: `docker run -it --gpus all sparseml_docker`
This will run the bash console in the container's `root` directory.

## Examples
Once inside the container, the `sparseml` can be used right away.

Note: RuntimeError: DataLoader worker (pid 1388) is killed by signal: Bus error. It is possible that dataloader's workers are out of shared memory. Please try to raise your shared memory limit.

### Example 1: Image Classification Pipeline:

```
sparseml.image_classification.train \ 
--recipe-path zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned85_quant-none-vnni \
--arch-key resnet50 \  
--pretrained pruned-moderate \   
--dataset imagenette \
--dataset-path dataset \     
--train-batch-size 4 \ 
--test-batch-size 8 
```

### Example 2: Transformers Question Answering Pipeline:

```python
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
