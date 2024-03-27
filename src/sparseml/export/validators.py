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

import glob
import logging
import os.path
from collections import OrderedDict
from pathlib import Path
from typing import Callable, List, Optional
from typing import OrderedDict as OrderedDictType
from typing import Union

import numpy
import onnx

from sparseml.export.export_data import InputsNames, LabelNames, OutputsNames
from sparseml.export.helpers import ONNX_MODEL_NAME, onnx_data_files
from sparsezoo.utils.numpy import load_numpy


__all__ = ["validate_correctness", "validate_structure"]

_LOGGER = logging.getLogger(__name__)


def validate_structure(
    target_path: Union[str, Path],
    deployment_directory_name: str,
    onnx_model_name: str,
    deployment_directory_files_mandatory: List[str],
    deployment_directory_files_optional: Optional[List[str]] = None,
):
    """
    Validates the structure of the targe_path by
    checking if the expected files, that should exist as a result
    of the export, are present.

    :param target_path: The directory where the exported files are stored.
    :param deployment_directory_name: The name of the deployment directory.
    :param onnx_model_name: The name of the ONNX model.
    :param deployment_directory_files_mandatory: The list of files that
        should be present in the deployment directory.
    :param deployment_directory_files_optional: The list of files that
        can be optionally present in the deployment directory.
    """
    deployment_directory_path = os.path.join(target_path, deployment_directory_name)

    validate_structure_external_data(
        deployment_directory_path, onnx_model_name=onnx_model_name
    )

    sample_files = {InputsNames, OutputsNames, LabelNames}

    # account for the potentially custom ONNX model name
    deployment_directory_files_mandatory = [
        onnx_model_name if file_name == ONNX_MODEL_NAME else file_name
        for file_name in deployment_directory_files_mandatory
    ]
    # obtain full paths
    deployment_directory_files_mandatory = {
        os.path.join(deployment_directory_path, file_name)
        for file_name in deployment_directory_files_mandatory
    }
    deployment_directory_files_optional = {
        os.path.join(deployment_directory_path, file_name)
        for file_name in deployment_directory_files_optional or []
    }

    # obtain full paths for the potential sample files
    optional_files = {
        os.path.join(target_path, name.basename.value) for name in sample_files
    }
    optional_files.update(deployment_directory_files_optional)

    missing_mandatory_files = check_file_presence(deployment_directory_files_mandatory)
    missing_optional_files = check_file_presence(optional_files)

    if missing_optional_files:
        for file_path in missing_optional_files:
            _LOGGER.warning(f"File {file_path} is missing.")

    if missing_mandatory_files:
        for file_path in missing_mandatory_files:
            raise FileNotFoundError(f"File {file_path} is missing.")


def validate_structure_external_data(
    deployment_directory_path: Union[str, Path], onnx_model_name: Union[str, Path]
):
    files_present = onnx_data_files(
        onnx_model_name.replace(".onnx", ".data"), deployment_directory_path
    )
    if files_present:
        _LOGGER.info(
            f"Exported model contains {len(files_present)} external data files"
        )


def check_file_presence(file_paths: List[str]) -> List[str]:
    """
    Check if the files exist in the given paths.

    :param file_paths: The list of paths to check.
        Paths can be either directories or files.
    :return The list of missing file paths.
    """
    missing_files = []
    for file_path in file_paths:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    return missing_files


def top_k_match(
    ground_truth: numpy.ndarray, prediction: numpy.ndarray, k: int = 2
) -> bool:
    """
    Checks if the top k predictions match the ground truth.

    :param ground_truth: The ground truth array.
    :param prediction: The prediction array.
    :param k: The number of top predictions to consider.
    """
    top_k_prediction = numpy.argsort(prediction.flatten())[-k:]
    top_k_ground_truth = numpy.argsort(ground_truth.flatten())[-k:]
    return numpy.all(top_k_prediction == top_k_ground_truth)


def validate_correctness(
    target_path: Union[str, Path],
    directory: Union[str, Path],
    onnx_model_name: str,
    validation_function: Callable[..., bool] = top_k_match,
) -> bool:
    """
    Validates the correctness of the exported ONNX model by
    running it on a set of sample inputs and comparing the
    resulting outputs using a validation function.

    :param target_path: The directory where the sample inputs and outputs are stored.
    :param directory: The directory where the ONNX model is stored.
    :param onnx_model_name: The name of the ONNX model.
    :param validation_function: The function that will be used to validate the outputs.
    :return: True if the validation passes, False otherwise.
    """
    try:
        import onnxruntime as ort
    except ImportError as err:
        raise ImportError(
            "The onnxruntime package is required for the correctness validation. "
            "Please install it using 'pip install sparseml[onnxruntime]'."
        ) from err

    sample_inputs_path = os.path.join(target_path, InputsNames.basename.value)
    sample_outputs_path = os.path.join(target_path, OutputsNames.basename.value)

    sample_inputs_files = sorted(glob.glob(os.path.join(sample_inputs_path, "*")))
    sample_outputs_files = sorted(glob.glob(os.path.join(sample_outputs_path, "*")))
    model_path = os.path.join(directory, onnx_model_name)
    expected_input_names = [
        inp.name for inp in onnx.load(model_path, load_external_data=False).graph.input
    ]
    session = ort.InferenceSession(model_path)

    validations = (
        []
    )  # stores boolean per sample pair (True if validation passes, False otherwise)

    for sample_input_file, sample_output_file in zip(
        sample_inputs_files, sample_outputs_files
    ):
        sample_input = load_numpy(sample_input_file)
        sample_output = load_numpy(sample_output_file)

        sample_input_with_batch_dim = OrderedDict(
            (key, numpy.expand_dims(value, 0)) for key, value in sample_input.items()
        )

        sample_input_with_batch_dim = _potentially_rename_input(
            sample_input_with_batch_dim, expected_input_names
        )

        outputs = session.run(None, sample_input_with_batch_dim)
        if isinstance(outputs, list):
            validations_sample = []
            for o1, o2 in zip(outputs, sample_output.values()):
                validations_sample.append(validation_function(o1, o2))
            validations.append(all(validations_sample))
        else:
            validations.append(validation_function(outputs, sample_output))

    if not all(validations):
        _LOGGER.error(
            f"Correctness validation failed for exported model: {onnx_model_name}. "
            "The model outputs match the expected outputs "
            f"only for {sum(validations)}/{len(validations)} samples "
            f"(according to the validation function: {validation_function.__name__}. "
            f"Some failures are expected in the case of quantized models, but not in "
            f"the case of non-quantized models. If in doubt, validate the performance "
            f"of the exported ONNX model using the NeuralMagic evaluation module."
        )
        return False

    _LOGGER.info(
        f"Successfully validated the exported model on all {len(validations)} samples."
    )
    return True


def _potentially_rename_input(
    sample_input_with_batch_dim: OrderedDictType[str, numpy.ndarray],
    expected_input_names: List[str],
) -> OrderedDictType[str, numpy.ndarray]:
    # if required, rename the input names of the sample to match
    # the input names of the model
    input_names = list(sample_input_with_batch_dim.keys())
    if set(input_names) != set(expected_input_names):
        return OrderedDict(
            zip(expected_input_names, sample_input_with_batch_dim.values())
        )
    return sample_input_with_batch_dim
