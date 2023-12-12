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
from pathlib import Path
from typing import List, Union

from sparseml.export.helpers import InputsNames, OutputsNames
from sparsezoo.inference import InferenceRunner
from sparsezoo.objects import File, NumpyDirectory


__all__ = ["validate_correctness", "validate_structure"]


def validate_structure(
    directory: Union[str, Path], deployment_directory_files: List[str]
):
    """
    Validates the structure of the exported ONNX model by
    checking if the required files are present.

    :param directory: The directory where the ONNX model is stored.
    :param deployment_directory_files: The list of files that should be present
        in the deployment directory.
    """
    for file in deployment_directory_files:
        if not os.path.exists(os.path.join(directory, file)):
            raise ValueError(
                f"File {file} is missing from the deployment directory {directory}"
            )


# TODO: Test this function with e2e tests
def validate_correctness(directory: Union[str, Path], onnx_model_name: str):
    """
    Validates the correctness of the exported ONNX model by
    running it on a set of sample inputs and comparing the
    resulting outputs with precomputed ground truth values.

    :param directory: The directory where the ONNX model is stored.
    :param onnx_model_name: The name of the ONNX model.
    """
    # TODO: During testing add a support for tar.gz scenario (potentially)
    sample_inputs_path = os.path.join(directory, InputsNames.basename.value)
    sample_outputs_path = os.path.join(directory, OutputsNames.basename.value)

    sample_inputs = NumpyDirectory(
        name=InputsNames.basename.value,
        files=os.listdir(sample_inputs_path),
        path=sample_inputs_path,
    )
    sample_outputs = NumpyDirectory(
        name=OutputsNames.basename.value,
        files=os.listdir(sample_outputs_path),
        path=sample_outputs_path,
    )
    onnx_model = File(
        name=onnx_model_name, path=os.path.join(directory, onnx_model_name)
    )

    runner = InferenceRunner(
        sample_inputs=sample_inputs,
        sample_outputs=sample_outputs,
        onnx_file=onnx_model,
    )

    runner.validate_with_onnx_runtime()
