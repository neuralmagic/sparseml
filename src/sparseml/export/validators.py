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

import logging
import os.path
from pathlib import Path
from typing import List, Union

from sparseml.export.export_data import InputsNames, LabelNames, OutputsNames
from sparseml.export.helpers import ONNX_MODEL_NAME
from sparsezoo.inference import InferenceRunner
from sparsezoo.objects import File, NumpyDirectory


__all__ = ["validate_correctness", "validate_structure"]

_LOGGER = logging.getLogger(__name__)


def validate_structure(
    target_path: Union[str, Path],
    deployment_directory_name: str,
    onnx_model_name: str,
    deployment_directory_files: List[str],
):
    """
    Validates the structure of the targe_path by
    checking if the expected files, that should exist as a result
    of the export, are present.

    :param target_path: The directory where the exported files are stored.
    :param deployment_directory_name: The name of the deployment directory.
    :param onnx_model_name: The name of the ONNX model.
    :param deployment_directory_files: The list of files that should be present
        in the deployment directory.
    """
    sample_files = {InputsNames, OutputsNames, LabelNames}

    deployment_directory_files = [
        onnx_model_name if file_name == ONNX_MODEL_NAME else file_name
        for file_name in deployment_directory_files
    ]

    mandatory_files = {
        os.path.join(target_path, deployment_directory_name, file_name)
        for file_name in deployment_directory_files
    }
    optional_files = {
        os.path.join(target_path, name.basename.value) for name in sample_files
    }
    check_file_presence(mandatory_files, mandatory=True)
    check_file_presence(optional_files, mandatory=False)


def check_file_presence(file_paths: List[str], mandatory: bool = False):
    """
    Check if the files exist in the given paths.

    :param file_paths: The list of paths to check.
        Paths can be either directories or files.
    :param mandatory: If True, raises an error if
        any of the files is missing. Otherwise,
        logs a warning.
    """
    for file_path in file_paths:
        if not os.path.exists(file_path):
            if mandatory:
                raise FileNotFoundError(f"File {file_path} is missing.")
            else:
                _LOGGER.warning(f"File {file_path} is missing.")


# TODO: Need to add few changes to sparsezoo to support this function
def validate_correctness(
    target_path: Union[str, Path], directory: Union[str, Path], onnx_model_name: str
):
    """
    Validates the correctness of the exported ONNX model by
    running it on a set of sample inputs and comparing the
    resulting outputs with precomputed ground truth values.

    :param target_path: The directory where the sample inputs and outputs are stored.
    :param directory: The directory where the ONNX model is stored.
    :param onnx_model_name: The name of the ONNX model.
    """
    # TODO: During testing add a support for tar.gz scenario (potentially)
    sample_inputs_path = os.path.join(target_path, InputsNames.basename.value)
    sample_outputs_path = os.path.join(target_path, OutputsNames.basename.value)

    sample_inputs = NumpyDirectory(
        name=InputsNames.basename.value,
        files=[
            File(name=file_name, path=os.path.join(sample_inputs_path, file_name))
            for file_name in os.listdir(sample_inputs_path)
        ],
        path=sample_inputs_path,
    )
    sample_outputs = NumpyDirectory(
        name=OutputsNames.basename.value,
        files=[
            File(name=file_name, path=os.path.join(sample_outputs_path, file_name))
            for file_name in os.listdir(sample_outputs_path)
        ],
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
