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

from sparsezoo.v2.helpers import setup_model_directory


__all__ = ["get_model_directory"]


def get_model_directory(
    output_dir: str,
    training_outputs_dir: str,
    model_file_torch: str,
) -> None:
    """
    Takes the `training_outputs_dir`
    (the directory where the pipeline saves its training artifacts),
    and saves the training artifacts to `output_dir` in the `ModelDirectory` structure.

    :param output_dir: The output path where the artifacts are saved
        (adhering to the in `ModelDirectory` structure)
    :param training_outputs_dir: The path to the existing directory
        with the saved training artifacts
    :param model_file_torch: name of the final .pth/.pt file to be saved
    """
    training_outputs_files = glob.glob(os.path.join(training_outputs_dir, "*"))
    logs_path = [
        file
        for file in training_outputs_files
        if os.path.basename(file).startswith("events.out.")
    ]
    for root_file in ["sample_inputs.tar.gz", "sample_outputs.tar.gz"]:
        root_file_path = os.path.join(training_outputs_dir, root_file)
        if not os.path.exists(root_file_path):
            raise ValueError(
                f"File {root_file_path} missing. To create this file, "
                "make sure that the training script is being ran with"
                "`--num_export_samples` argument."
            )

    model_onnx_path = os.path.join(
        training_outputs_dir, "weights", model_file_torch.replace(".pt", ".onnx")
    )
    if not os.path.exists(model_onnx_path):
        raise ValueError(
            f"File {model_onnx_path} missing. To create this file, "
            "make sure that the `export` script (for exporting "
            "yolo model) has been evoked."
        )

    setup_model_directory(
        output_dir=output_dir,
        training=[os.path.join(training_outputs_dir, "weights", model_file_torch)],
        deployment=[model_onnx_path],
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
