# SparseML docker image
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
To build and launch this image with the tag `sparseml_docker`, run from the root directory: `docker build -t sparseml_docker . && docker run -it sparseml_docker ${python_command},` for example`docker build -t sparseml_docker . && docker container run --gpus all --shm-size=256m -it sparseml_docker sparseml.transformers.train.token_classification \
  --model_name_or_path zoo:nlp/masked_language_modeling/obert-base/pytorch/huggingface/wikipedia_bookcorpus/pruned90-none \
  --recipe zoo:nlp/token_classification/obert-base/pytorch/huggingface/conll2003/pruned90_quant-none \
  --distill_teacher zoo:nlp/token_classification/obert-base/pytorch/huggingface/conll2003/base-none \
  --dataset_name conll2003 \
  --output_dir sparse_bert-token_classification_conll2003 \
  --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --preprocessing_num_workers 6 \
  --do_train --do_eval --evaluation_strategy epoch --fp16 --seed 29204  \
  --save_strategy epoch --save_total_limit 1`  

If you want to use a specific branch from sparseml you can use the `GIT_CHECKOUT` build arg:
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
