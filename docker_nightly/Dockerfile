ARG SOURCE=ghcr.io/neuralmagic/cuda-python3.8

FROM $SOURCE

RUN python3.8 -m pip install --upgrade pip \
    && python3.8 -m pip install --upgrade setuptools \
    && python3.8 -m pip install --no-cache-dir "sparseml-nightly[onnxruntime,torchvision,transformers,yolov5,ultralytics]"

HEALTHCHECK CMD python3.8 -c 'import sparseml'
RUN python3.8 -m pip list | grep sparseml
CMD bash