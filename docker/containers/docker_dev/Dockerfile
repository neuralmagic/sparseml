ARG SOURCE=ghcr.io/neuralmagic/cuda-python3.8
ARG BRANCH

FROM $SOURCE

ARG BRANCH

RUN python3.8 -m pip install --upgrade pip \
    && python3.8 -m pip install --upgrade setuptools \
    && git clone https://github.com/neuralmagic/sparseml.git --depth 1 --single-branch -b ${BRANCH} \
    && python3.8 -m pip install -e "./sparseml[dev]"

HEALTHCHECK CMD python3.8 -c 'import sparseml'
RUN python3.8 -m pip list | grep sparseml
CMD bash