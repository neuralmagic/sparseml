
import numpy as np
import glob
import os
from deepsparse import Pipeline
import matplotlib.pyplot as plt

path_inputs = "/home/ubuntu/damian/sparseml/exported/deployment/sample_inputs"
path_outputs_torch = "/home/ubuntu/damian/sparseml/exported/deployment/sample_outputs_torch"
path_outputs_ort = "/home/ubuntu/damian/sparseml/exported/deployment/sample_outputs_ort"
onnx_model = "/home/ubuntu/damian/sparseml/exported/deployment/DetectionModel.onnx"
onnx_model_pretrained = "/home/ubuntu/damian/sparseml/yolov8n.onnx"

pipeline = Pipeline.create(task = "yolov8", model_path = onnx_model_pretrained, class_names="coco")
inputs = glob.glob(os.path.join(path_inputs, "*"))
inputs.sort()
outputs_torch = glob.glob(os.path.join(path_outputs_torch, "*"))
outputs_torch.sort()
outputs_ort = glob.glob(os.path.join(path_outputs_ort, "*"))
outputs_ort.sort()
for output_torch, output_ort in zip(outputs_torch, outputs_ort):
    output_torch = np.load(output_torch, allow_pickle=True)
    output_torch = output_torch.f.arr_0
    output_ort = np.load(output_ort, allow_pickle=True)
    output_ort = output_ort.f.arr_0
    assert np.allclose(output_ort, output_torch, atol=1e-04)

for input in inputs:
    input = np.load(input, allow_pickle=True)
    input = input.f.arr_0

    images = (input * 255).astype(np.uint8)

    plt.imshow(images[0].transpose(1, 2, 0))
    plt.show()

    out = pipeline(images=images)
    print(out.labels)
    pass



