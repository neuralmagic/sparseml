ARG TORCH_VERSION=1.9.1
ARG TORCHVISION_VERSION=0.10.1
ARG CUDA_VERSION

# Setup the base image & install dependencies
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu18.04 as base
RUN : \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        software-properties-common \
        git-all \
    && add-apt-repository -y ppa:deadsnakes \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3.8-venv \
        python3-pip \
        python3-dev \
        build-essential libssl-dev libffi-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Activate venv
RUN python3.8 -m venv /venv
ENV PATH="venv/bin:$PATH"

FROM base as cuda-10.2
ARG TORCH_VERSION
ARG TORCHVISION_VERSION
RUN pip3 install --upgrade torch==${TORCH_VERSION}+cu102 torchvision==${TORCHVISION_VERSION}+cu102  -f https://download.pytorch.org/whl/torch_stable.html

FROM base as cuda-11.1
ARG TORCH_VERSION
ARG TORCHVISION_VERSION
RUN pip3 install --upgrade pip torch==${TORCH_VERSION}+cu111 torchvision==${TORCHVISION_VERSION}+cu111 -f https://download.pytorch.org/whl/torch_stable.html

FROM cuda-$CUDA_VERSION as target

# Install SparseML
RUN pip3 install sparseml
