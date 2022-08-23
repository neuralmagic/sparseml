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

import os
import shutil
import tempfile
from pathlib import Path

import pytest

from sparseml.yolov5.helpers import save_zoo_directory
from sparsezoo import Model


@pytest.mark.parametrize(
    "stub",
    [
        "zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned_quant-aggressive_94",  # noqa E501
    ],
)
def test_save_zoo_directory(stub):
    path_to_training_outputs = tempfile.mktemp()
    save_dir = tempfile.mktemp()

    zoo_model = Model(stub, path_to_training_outputs)

    zoo_model.deployment.default.download()
    zoo_model.model_card.download()

    sample_inputs = zoo_model.sample_inputs
    sample_inputs.download()

    sample_outputs = zoo_model.sample_outputs
    sample_outputs["framework"].download()

    # create weights directory
    weights_directory = os.path.join(path_to_training_outputs, "weights")
    os.mkdir(weights_directory)

    # prepare files to be copied over
    model_path = zoo_model.training.default.get_file("model.pt").path
    onnx_model_path = zoo_model.onnx_model.path

    # copy
    for path in [model_path, onnx_model_path]:
        if os.path.isdir(path):
            shutil.copytree(
                path, os.path.join(weights_directory, os.path.basename(path))
            )
        else:
            shutil.copyfile(
                path, os.path.join(weights_directory, os.path.basename(path))
            )

    # create dummy events
    Path(os.path.join(path_to_training_outputs, "events.out.dummy")).touch()

    save_zoo_directory(
        output_dir=save_dir,
        training_outputs_dir=path_to_training_outputs,
        model_file_torch="model.pt",
    )
    new_zoo_model = Model(save_dir)
    assert new_zoo_model.validate(minimal_validation=True, validate_onnxruntime=False)

    shutil.rmtree(path_to_training_outputs)
    shutil.rmtree(save_dir)
