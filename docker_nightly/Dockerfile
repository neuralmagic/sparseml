ARG SOURCE=ghcr.io/neuralmagic/cuda-python3.8

FROM $SOURCE

RUN pip install ---upgrade pip && pip install --upgrade setuptools && pip install --no-cache-dir "sparseml-nightly[onnxruntime,torchvision,transformers,yolov5,ultralytics]"

HEALTHCHECK CMD python3.8 -c 'import sparseml'
RUN pip list | grep sparseml
CMD bash