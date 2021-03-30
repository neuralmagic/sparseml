<!--
Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Example YOLO Model Server and Client Using DeepSparse Flask

To illustrate how the DeepSparse engine can be used for YOLO model deployments, this directory
contains a sample model server and client. 

The server uses Flask to create an app with the DeepSparse Engine hosting a
compiled YOLOv3 model.
The client can make requests into the server returning object detection results for given images.


## Installation

Similarly to the SparseML integration, dependencies can be installed via `pip` and the files in
this self-contained example can be copied directly into the `ultralytics/yolov5` for execution.

If both repositories are already cloned, you may skip that step.

```bash
# clone
git clone https://github.com/ultralytics/yolov5.git
git clone https://github.com/neuralmagic/sparseml.git

# copy DeepSparse python files
cp sparseml/integrations/ultralytics/deepsparse/*.py yolov5
cd yolov5

# install dependencies
pip install -r requirements.txt
pip install deepsparse sparseml flask flask-cors
```

## Execution

### Server

First, start up the host `server.py` with your model of choice, SparseZoo stubs are
also supported.

Example command:
```bash
python server.py \
    zoo:cv/detection/yolo_v3-spp/pytorch/ultralytics/coco/pruned_quant-aggressive_94 \
    --quantized-inputs
```

You can leave that running as a detached process or in a spare terminal.

This starts a Flask app with the DeepSparse Engine as the inference backend, accessible at `http://0.0.0.0:5543` by default.

The app exposes HTTP endpoints at:
- `/info` to get information about the compiled model
- `/predict` to send images to the model and receive detected in response.
    The number of images should match the compiled model's batch size.

For a full list of options, run `python server.py -h`.

Currently, the server is set to do pre-processing for the yolov3-spp
model, if other models are used, the image shape, output shapes, and
anchor grids should be updated. 

### Client

`client.py` provides a `YoloDetectionClient` object to make requests to the server easy.
The file is self documented.  See example usage below:

```python
from client import YoloDetectionClient

remote_model = YoloDetectionClient()
image_path = "/PATH/TO/EXAMPLE/IMAGE.jpg"

model_outputs = remote_model.detect(image_path)
```
