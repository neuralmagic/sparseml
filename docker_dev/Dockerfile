ARG SOURCE=ghcr.io/neuralmagic/cuda-python3.8
ARG BRANCH

FROM $SOURCE

ARG BRANCH

RUN python3.8 -m pip install --upgrade pip \
    && python3.8 -m pip install --upgrade setuptools \
    && apt-get clean

RUN git clone https://github.com/neuralmagic/sparseml.git --depth 1 -b ${BRANCH} \
    && python3.8 -m pip install -e "./sparseml[dev,onnxruntime,torchvision,transformers]" \
    && apt-get clean

HEALTHCHECK CMD python3.8 -c 'import sparseml'
RUN python3.8 -m pip list | grep sparseml
CMD bash