import onnxruntime
import numpy as np
import glob
import os
from deepsparse import Pipeline
import matplotlib.pyplot as plt

path_inputs = "/home/ubuntu/damian/sparseml/exported/deployment/sample_inputs"
path_outputs = "/home/ubuntu/damian/sparseml/exported/deployment/sample_outputs"
onnx_model = "/home/ubuntu/damian/sparseml/exported/deployment/DetectionModel.onnx"
onnx_model_pretrained = "/home/ubuntu/damian/sparseml/yolov8n.onnx"

pipeline = Pipeline.create(task = "yolov8", model_path = onnx_model_pretrained, class_names="coco")
ort_session = onnxruntime.InferenceSession(onnx_model)
inputs = glob.glob(os.path.join(path_inputs, "*"))
inputs.sort()
outputs = glob.glob(os.path.join(path_outputs, "*"))
outputs.sort()
for ins, outs in zip(inputs, outputs):
    in_numpy = np.load(ins, allow_pickle=True)
    in_numpy = in_numpy.f.arr_0
    out_numpy = np.load(outs, allow_pickle=True)
    out_numpy = out_numpy.f.arr_0
    ort_inputs = {ort_session.get_inputs()[0].name: in_numpy}
    ort_outs = ort_session.run(None, ort_inputs)
    preds, *_ = ort_outs
    assert np.allclose(preds, out_numpy, atol=1e-04)
    images = (in_numpy[0] * 255).astype(np.uint8).transpose(1,2,0)
    import numpy as np
    import matplotlib.pyplot as plt
    plt.imshow(images)
    plt.show()
    out = pipeline(images=images)
    print(out.labels)
    pass



