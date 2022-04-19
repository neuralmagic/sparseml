## SparseML docker image
This directory contains the Dockerfile to create minimal SparseML docker image.
This image is based off the official NVIDIA development Ubuntu 18.04.5 LTS image (with CUDA support).

In order to build this image, run from the root directory:
- for compute platform CUDA 10.2: `docker build --build-arg CUDA_VERSION=10.2 -t sparseml_docker . && docker run -it --gpu all sparseml_docker`
- for compute platform CUDA 11.3: `docker build --build-arg CUDA_VERSION=11.3.1 -t sparseml_docker . && docker run -it --gpu all sparseml_docker
`
