ARG TORCH_VERSION=1.9.1
ARG TORCHVISION_VERSION=0.10.1
ARG CUDA_VERSION=10.2
ARG VERSION
ARG DEPS="all"
ARG VENV="/venv"
ARG BRANCH

# Setup the base image & install dependencies
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu18.04 as base
# As of 05/05/22 nvidia images are broken. Two lines below are a temporary fix.
# Source: https://github.com/NVIDIA/nvidia-docker/issues/1632

ARG VENV

# metainformation
LABEL org.opencontainers.image.version = "1.0.1"
LABEL org.opencontainers.image.authors = "Neuralmagic, Inc."
LABEL org.opencontainers.image.source = "https://github.com/neuralmagic/sparseml"
LABEL org.opencontainers.image.licenses = "Apache License 2.0"

RUN if [ -f /etc/apt/sources.list.d/cuda.list ] ; then rm /etc/apt/sources.list.d/cuda.list ; fi
RUN if [ -f /etc/apt/sources.list.d/nvidia-ml.list ] ; then rm /etc/apt/sources.list.d/nvidia-ml.list ; fi

RUN set -Eeuxo \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        software-properties-common \
        git-all \
    && add-apt-repository -y ppa:deadsnakes \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3.8-dev \
        python3.8-venv \
        build-essential libssl-dev libffi-dev \
        ffmpeg libsm6 libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Activate venv
RUN python3.8 -m venv $VENV
ENV PATH="${VENV}/bin:$PATH"

FROM base as cuda-10.2
ARG TORCH_VERSION
ARG VENV
ARG TORCHVISION_VERSION
RUN $VENV/bin/pip install --upgrade pip torch==${TORCH_VERSION}+cu102 torchvision==${TORCHVISION_VERSION}+cu102  -f https://download.pytorch.org/whl/torch_stable.html

FROM base as cuda-11.1.1
ARG TORCH_VERSION
ARG VENV
ARG TORCHVISION_VERSION
RUN $VENV/bin/pip install --upgrade pip torch==${TORCH_VERSION}+cu111 torchvision==${TORCHVISION_VERSION}+cu111 -f https://download.pytorch.org/whl/torch_stable.html

FROM cuda-$CUDA_VERSION as cuda_builder
ARG VENV
RUN $VENV/bin/pip install --upgrade setuptools wheel
ARG BRANCH
ARG VERSION
RUN \
    if [ -z "${VERSION}" ] ; then  \
    echo VERSION not provided && \
      if [ -z "${BRANCH}" ]; then  \
        echo cloning main \
          && git clone https://github.com/neuralmagic/sparseml.git --depth 1 -b main; \
      else echo cloning $BRANCH \
          && git clone https://github.com/neuralmagic/sparseml.git --depth 1 -b $BRANCH; \
      fi; \
    else echo VERSION is $VERSION \
      &&  git clone https://github.com/neuralmagic/sparseml.git --depth 1 -b ${VERSION}; \
    fi;


FROM cuda_builder AS container_branch_base
ARG VENV
ENV PATH="${VENV}/bin:$PATH"
ENV PIP_DEFAULT_TIMEOUT=200
ARG SPARSEML_INSTALLABLE=./sparseml
RUN $VENV/bin/pip install --no-cache-dir "${SPARSEML_INSTALLABLE}"


FROM container_branch_base as container_branch_all
ARG VENV
ARG SPARSEML_INSTALLABLE=./sparseml[onnxruntime,torchvision,ultralytics]
RUN $VENV/bin/pip install --no-cache-dir "${SPARSEML_INSTALLABLE}"
RUN sparseml.transformers.question_answering --help \
    && sparseml.yolov5.train --help \
    && sparseml.ultralytics.train --help

FROM container_branch_base as container_branch_transformers
RUN sparseml.transformers.question_answering --help


FROM container_branch_base as container_branch_torchvision
ARG VENV
ARG SPARSEML_INSTALLABLE=./sparseml[torchvision]
RUN $VENV/bin/pip install --no-cache-dir "${SPARSEML_INSTALLABLE}"

FROM container_branch_base as container_branch_ultralytics
ARG VENV
ARG SPARSEML_INSTALLABLE=./sparseml[onnxruntime,ultralytics]
RUN $VENV/bin/pip install --no-cache-dir "${SPARSEML_INSTALLABLE}"
RUN sparseml.ultralytics.train --help


FROM container_branch_base as container_branch_yolov5
ARG VENV
ARG SPARSEML_INSTALLABLE=./sparseml
RUN $VENV/bin/pip install --no-cache-dir "${SPARSEML_INSTALLABLE}"
RUN sparseml.yolov5.train --help


FROM container_branch_base as container_branch_dev
ARG VENV
ARG SPARSEML_INSTALLABLE=./sparseml[dev]
RUN $VENV/bin/pip install --no-cache-dir "${SPARSEML_INSTALLABLE}"


FROM container_branch_${DEPS} AS build
RUN echo Build complete, going onto prod

FROM base as prod
ARG VENV
COPY --from=build $VENV $VENV
ENV PATH="${VENV}/bin:$PATH"
HEALTHCHECK CMD python -c 'import sparseml'
ENTRYPOINT bash

