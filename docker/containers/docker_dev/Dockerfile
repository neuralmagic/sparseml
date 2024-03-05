ARG SOURCE=ghcr.io/neuralmagic/cuda-python3.8
# how were these versions selected before?
ARG TORCH_VERSION=1.10.2
ARG TORCHVISION_VERSION=0.11.3
ARG CUDA=113
ARG BRANCH

FROM $SOURCE

ARG BRANCH

RUN python3.8 -m pip install --upgrade pip \
    && python3.8 -m pip install --upgrade setuptools

ARG CUDA
ARG TORCH_VERSION
ARG TORCHVISION_VERSION

RUN python3.8 -m pip install torch==${TORCH_VERSION}+cu${CUDA} torchvision==${TORCHVISION_VERSION}+cu${CUDA} -f https://download.pytorch.org/whl/torch_stable.html \
    && git clone https://github.com/neuralmagic/sparseml.git --depth 1 --single-branch -b ${BRANCH} \
    && python3.8 -m pip install -e "./sparseml[dev]"

HEALTHCHECK CMD python3.8 -c 'import sparseml'
RUN python3.8 -m pip list | grep sparseml
CMD bash