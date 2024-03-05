ARG SOURCE=ghcr.io/neuralmagic/cuda-python3.8
ARG TORCH_VERSION=1.10.1
ARG TORCHVISION_VERSION=0.11.3
ARG CUDA=113

FROM $SOURCE

RUN python3.8 -m pip install --upgrade pip \
    && python3.8 -m pip install --upgrade setuptools

ARG CUDA 
ARG TORCH_VERSION
ARG TORCHVISION_VERSION

RUN python3.8 -m pip install torch==${TORCH_VERSION}+cu${CUDA} torchvision==${TORCHVISION_VERSION}+cu${CUDA} -f https://download.pytorch.org/whl/torch_stable.html \
    && python3.8 -m pip install --no-cache-dir "sparseml-nightly[onnxruntime,torchvision,transformers,yolov5,ultralytics]" 

HEALTHCHECK CMD python3.8 -c 'import sparseml'
RUN python3.8 -m pip list | grep sparseml
CMD bash