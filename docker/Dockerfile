ARG TORCH_VERSION=1.9.1
ARG TORCHVISION_VERSION=0.10.1
ARG CUDA_VERSION

# Setup the base image & install dependencies
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu18.04 as base
# As of 05/05/22 nvidia images are broken. Two lines below are a temporary fix.
# Source: https://github.com/NVIDIA/nvidia-docker/issues/1632

# metainformation
LABEL org.opencontainers.image.version = "1.0.1"
LABEL org.opencontainers.image.authors = "Neuralmagic, Inc."
LABEL org.opencontainers.image.source = "https://github.com/neuralmagic/sparseml"
LABEL org.opencontainers.image.licenses = "Apache License 2.0"

RUN if [ -f /etc/apt/sources.list.d/cuda.list ] ; then rm /etc/apt/sources.list.d/cuda.list ; fi
RUN if [ -f /etc/apt/sources.list.d/nvidia-ml.list ] ; then rm /etc/apt/sources.list.d/nvidia-ml.list ; fi

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

RUN pip3 install --upgrade setuptools wheel

ARG GIT_CHECKOUT
# Install SparseML
# if $GIT_CHECKOUT is not specified - just install from pypi
RUN if [ -z "${GIT_CHECKOUT}" ] ; then pip3 install --no-cache-dir --upgrade sparseml[dev,torchvision] ; fi

# if $GIT_CHECKOUT is specified - clone, checkout $GIT_CHECKOUT, and install with -e
RUN if [ -n "${GIT_CHECKOUT}" ] ; then git clone https://github.com/neuralmagic/sparseml.git --depth 1 -b $GIT_CHECKOUT; fi
RUN if [ -n "${GIT_CHECKOUT}" ] ; then pip3 install --no-cache-dir --upgrade -e "./sparseml[dev, torchvision]" ; fi
