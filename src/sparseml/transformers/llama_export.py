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

import argparse

from sparseml.transformers.export import MODEL_ONNX_NAME, export


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a trained LLAMA model to an ONNX file. This step should be "
        "completed priror to running KV Cache Injection."
    )

    parser.add_argument(
        "--model_path",
        required=True,
        type=str,
        help=(
            "Path to directory where model files for weights, config, and "
            "tokenizer are stored"
        ),
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=384,
        help="Sequence length to use. Default is 384. Can be overwritten later",
    )
    parser.add_argument(
        "--onnx_file_name",
        type=str,
        default=MODEL_ONNX_NAME,
        help=(
            "Name for exported ONNX file in the model directory. "
            "Default and recommended value for pipeline "
            f"compatibility is {MODEL_ONNX_NAME}"
        ),
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help=("Set flag to allow custom models in HF-transformers"),
    )

    return parser.parse_args()


def main():
    args = _parse_args()
    export(
        task="text-generation",
        model_path=args.model_path,
        sequence_length=args.sequence_length,
        no_convert_qat=True,
        finetuning_task=None,
        onnx_file_name=args.onnx_file_name,
        num_export_samples=0,
        trust_remote_code=args.trust_remote_code,
        data_args=None,
        one_shot=None,
    )


if __name__ == "__main__":
    main()
