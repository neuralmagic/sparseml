"""
Script to convert an ONNX model's intializer's with sparse values to sparse initializers or
model with sparse initializers to only dense initializers.

##########
Command help:
usage: model_convert_sparse_tensors.py [-h] --onnx-file-path ONNX_FILE_PATH
                                       [--output-file-path OUTPUT_FILE_PATH]
                                       [--sparsity-threshold SPARSITY_THRESHOLD]
                                       [--convert-to-dense]
                                       [--skip-model-check]

convert an ONNX model's intializer's with sparse values to sparse initializers
or model with sparse initializers to only dense initializers

optional arguments:
  -h, --help            show this help message and exit
  --onnx-file-path ONNX_FILE_PATH
                        Path to the local onnx file to convert
  --output-file-path OUTPUT_FILE_PATH
                        Path to save the converted model. Default will be the
                        same file path as the input with '-sparse' or '-dense'
                        appended to the file name
  --sparsity-threshold SPARSITY_THRESHOLD
                        Minimum sparsity of a tensor to be converted to a
                        sparse tensor. Default is 0.6
  --convert-to-dense    set flag to convert a model with sparse initializers
                        to use dense initializers
  --skip-model-check    set flag to skip running the ONNX model checker before
                        saving a model

##########
Example converting a model to use sparse initializers:
python3 scripts/onnx/model_convert_sparse_tensors.py \
    --onnx-file-path /PATH/TO/MODEL.onnx

Example converting a model with sparse initializers to use dense initializers:
python3 scripts/onnx/model_convert_sparse_tensors.py \
    --onnx-file-path /PATH/TO/MODEL.onnx \
    --convert-to-dense
"""


import argparse
import onnx

from neuralmagicML import get_main_logger
from neuralmagicML.onnx.utils import (
    convert_model_initializers_to_sparse,
    convert_sparse_initializers_to_dense,
)

LOGGER = get_main_logger()


def parse_args():
    parser = argparse.ArgumentParser(
        description="convert an ONNX model's intializer's with sparse values to sparse"
        " initializers or model with sparse initializers to only dense initializers"
    )
    parser.add_argument(
        "--onnx-file-path",
        type=str,
        required=True,
        help="Path to the local onnx file to convert",
    )
    parser.add_argument(
        "--output-file-path",
        type=str,
        default="",
        help="Path to save the converted model. Default will be the same file path"
        " as the input with '-sparse' or '-dense' appended to the file name",
    )
    parser.add_argument(
        "--sparsity-threshold",
        type=float,
        default=0.6,
        help="Minimum sparsity of a tensor to be converted to a sparse tensor."
        " Default is 0.6",
    )
    parser.add_argument(
        "--convert-to-dense",
        action="store_true",
        help="set flag to convert a model with sparse initializers to use dense "
        "initializers",
    )
    parser.add_argument(
        "--skip-model-check",
        action="store_true",
        help="set flag to skip running the ONNX model checker before saving a model",
    )

    return parser.parse_args()


def main(args):
    model = onnx.load(args.onnx_file_path)

    if args.convert_to_dense:
        LOGGER.info("converting model to dense")
        convert_sparse_initializers_to_dense(model, inplace=True)
    else:
        LOGGER.info("converting model to sparse")
        convert_model_initializers_to_sparse(
            model, sparsity_threshold=args.sparsity_threshold, inplace=True
        )

    if not args.skip_model_check:
        try:
            LOGGER.info("checking converted model with onnx.checker.check_model")
            onnx.checker.check_model(model)
            LOGGER.info("check passed")
        except Exception as e:
            raise e

    output_path = (
        args.output_file_path
        if args.output_file_path
        else "{}-{}.onnx".format(
            args.onnx_file_path.split(".onnx")[0],
            "dense" if args.convert_to_dense else "sparse",
        )
    )
    onnx.save(model, output_path)
    LOGGER.info("converted model saved to {}".format(output_path))


if __name__ == "__main__":
    args_ = parse_args()
    main(args_)
