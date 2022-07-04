# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os.path

from sparsezoo.v2.model_directory import ModelDirectory
from src.sparseml.yolov5.helpers import get_model_directory


# where the training artifacts are
path_to_training_outputs = "/home/damian/yolo/yolov5_runs/train/exp5"
model_file_torch = "last.pt"
save_dir = "/home/damian/my_model_dir"

# create model card if none exists
model_card_path = os.path.join(path_to_training_outputs, "model.md")
if not os.path.exists(model_card_path):
    open(model_card_path, "w").close()


get_model_directory(
    output_dir=save_dir,
    training_outputs_dir=path_to_training_outputs,
    model_file_torch=model_file_torch,
)

# print the resulting directory
start_path = save_dir
for path, dirs, files in os.walk(start_path):
    for filename in files:
        print(os.path.join(path, filename))

# this will fail in a controlled manner (the model card is empty)
assert ModelDirectory.from_directory(save_dir)
