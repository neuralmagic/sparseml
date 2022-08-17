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

"""
Helper variables and functions for integrating SparseML with huggingface/transformers
flows
"""
import logging
import os
from typing import Optional

from sparsezoo.v2.helpers import setup_model_directory


__all__ = ["RECIPE_NAME", "get_model_directory"]


RECIPE_NAME = "recipe.yaml"
TRAINING_FILES = [
    "eval_results.json",
    "training_args.bin",
    "vocab.txt",
    "all_results.json",
    "trainer_state.json",
    "special_tokens_map.json",
    "train_results.json",
    RECIPE_NAME,
    "pytorch_model.bin",
    "config.json",
    "tokenizer_config.json",
    "tokenizer.json",
]
DEPLOYMENT_FILES = ["tokenizer.json", "config.json", "model.onnx"]


def get_model_directory(
    output_dir: str,
    training_outputs_dir: str,
    logs_path: Optional[str] = None,
    training_files: str = TRAINING_FILES,
    deployment_files: str = DEPLOYMENT_FILES,
):
    """
    Takes the `training_outputs_dir`
    (the directory where the pipeline saves its training artifacts),
    and saves the training artifacts to `output_dir` as a sparsezoo Model class object.

    :param output_dir: The output path where the artifacts are saved
        (adhering to the structure of sparsezoo Model class object)
    :param training_outputs_dir: The path to the existing directory
        with the saved training artifacts
    :param logs_path: Optional directory where the training logs reside
    :param training_files: List with files that belong to the `training` directory
    :param deployment_files: List with files that belong to the `deployment` directory
    """
    for root_file in ["sample_inputs", "sample_outputs"]:
        root_file_path = os.path.join(training_outputs_dir, root_file)
        if not os.path.exists(root_file_path):
            logging.warning(
                f"File {root_file_path} missing. To create this file, "
                "make sure that the export script is being ran with"
                "`--num_export_samples` argument."
            )

    model_onnx_path = os.path.join(training_outputs_dir, "model.onnx")
    if not os.path.exists(model_onnx_path):
        raise ValueError(
            f"File {model_onnx_path} missing. To create this file, "
            "make sure that the `export` script (for exporting "
            "transformer models) has been evoked."
        )

    setup_model(
        output_dir=output_dir,
        training=[
            os.path.join(training_outputs_dir, file_name)
            for file_name in training_files
        ],
        deployment=[
            os.path.join(training_outputs_dir, file_name)
            for file_name in deployment_files
        ],
        onnx_model=os.path.join(training_outputs_dir, "model.onnx"),
        sample_inputs=os.path.join(training_outputs_dir, "sample_inputs"),
        sample_outputs=os.path.join(training_outputs_dir, "sample_outputs"),
        model_card=os.path.join(training_outputs_dir, "model.md"),
        logs=logs_path,
        sample_labels=None,
        sample_originals=None,
        analysis=None,
        benchmarks=None,
        eval_results=None,
        recipes=None,
    )
    logging.info(f"Created sparsezoo Model directory locally in {output_dir}")
