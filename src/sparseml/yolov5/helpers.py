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
import os
import shutil

from sparsezoo import setup_model


__all__ = ["save_zoo_directory"]

MODEL_ONNX_NAME = "model.onnx"


def save_zoo_directory(
    output_dir: str,
    training_outputs_dir: str,
    model_file_torch: str,
) -> None:
    """
    Takes the `training_outputs_dir`
    (the directory where the pipeline saves its training artifacts),
    and saves the training artifacts to `output_dir` as a sparsezoo Model class object.

    :param output_dir: The output path where the artifacts are saved
        (adhering to the structure of sparsezoo Model class object)
    :param training_outputs_dir: The path to the existing directory
        with the saved training artifacts
    :param model_file_torch: name of the final .pth/.pt file to be saved
    """
    logs_path = [
        file
        for file in glob.glob(os.path.join(training_outputs_dir, "*"))
        if os.path.basename(file).startswith("events.out.")
    ]
    for root_file in ["sample_inputs.tar.gz", "sample_outputs.tar.gz"]:
        root_file_path = os.path.join(training_outputs_dir, root_file)
        if not os.path.exists(root_file_path):
            logging.warning(
                f"File {root_file_path} missing. To create this file, "
                "make sure that the export script has been ran with"
                "`--num_export_samples` argument."
            )
    model_onnx_path = os.path.join(
        training_outputs_dir, "weights", model_file_torch.replace(".pt", ".onnx")
    )
    deployment_path = os.path.join(training_outputs_dir, "deployment")

    for path in [model_onnx_path, deployment_path]:
        if not os.path.exists(path):
            raise ValueError(
                f"File {path} missing. To create this file, "
                "make sure that the `export` script (for exporting "
                "yolo model) has been evoked."
            )
    _assert_correct_model_onnx_name(model_onnx_path)
    _assert_correct_model_onnx_name(deployment_path)

    setup_model(
        output_dir=output_dir,
        training=[os.path.join(training_outputs_dir, "weights", model_file_torch)],
        deployment=deployment_path,
        onnx_model=model_onnx_path,
        sample_inputs=os.path.join(training_outputs_dir, "sample_inputs.tar.gz"),
        sample_outputs=os.path.join(training_outputs_dir, "sample_outputs.tar.gz"),
        model_card=os.path.join(training_outputs_dir, "model.md"),
        logs=logs_path,
        sample_labels=None,
        sample_originals=None,
        analysis=None,
        benchmarks=None,
        eval_results=None,
        recipes=None,
    )
    logging.info(f"Created `ModelDirectory` folder locally in {output_dir}")


def _assert_correct_model_onnx_name(onnx_file_or_parent_directory_path: str):
    # get a pointer to a single onnx file
    # (either direct path to the onnx file or to its parent directory)
    # and rename it to MODEL_ONNX_NAME if necessary
    if os.path.isdir(onnx_file_or_parent_directory_path):
        parent_dir = onnx_file_or_parent_directory_path
        parent_directory_files = os.listdir(parent_dir)
        onnx_file_name = [
            file_name
            for file_name in parent_directory_files
            if file_name.endswith(".onnx")
        ]
        if len(onnx_file_name) != 1:
            raise ValueError(
                f"Expected to find only one .onnx file inside the {parent_dir}. "
                f"However, found {len(onnx_file_name)} .onnx files"
            )
        onnx_file_path = os.path.join(parent_dir, onnx_file_name[0])
    else:
        onnx_file_path = onnx_file_or_parent_directory_path

    if not os.path.basename(onnx_file_path) == MODEL_ONNX_NAME:
        target_onnx_file_path = os.path.join(
            os.path.dirname(onnx_file_path), MODEL_ONNX_NAME
        )
        shutil.move(onnx_file_path, target_onnx_file_path)
