"""
Onnx model calibration and quantization script to support
static and dynamic post training model quantization.


Command help:
usage: model_quantize_post_training.py [-h] --onnx-file ONNX_FILE --data-glob
                                       DATA_GLOB --output-model-path
                                       OUTPUT_MODEL_PATH [--op-types OP_TYPES]
                                       [--exclude-nodes EXCLUDE_NODES]
                                       [--include-nodes INCLUDE_NODES]
                                       [--augmented-model-path AUGMENTED_MODEL_PATH]
                                       [--static STATIC] [--symmetric-weight]
                                       [--force-fusions]

Calibrate and Quantize an Onnx model

optional arguments:
  -h, --help            show this help message and exit
  --onnx-file ONNX_FILE
                        File path to onnx model to quantize
  --data-glob DATA_GLOB
                        Glob pattern to grab sample data files to feed through
                        the model, sample data must be numpy files with one
                        model input per file as either an array in an npy file
                        or dictionary in an npz file, defaults to random
  --output-model-path OUTPUT_MODEL_PATH
                        File path to where the quantized model should be
                        written
  --op-types OP_TYPES   Comma delimited operator types to be calibrated and
                        quantized
  --exclude-nodes EXCLUDE_NODES
                        Comma delimited operator names that should not be
                        quantized
  --include-nodes INCLUDE_NODES
                        Comma delimited operator names force to be quantized
  --augmented-model-path AUGMENTED_MODEL_PATH
                        Save augmented model to this file for verification
                        purpose
  --static STATIC       Use static quantization. (Default is static)
  --symmetric-weight    Use symmetric weight quantization. Default is False
  --force-fusions       Set to force fusions in quantization. Default is False
  --skip-extra-optimiation
                        Set to not run extra optimizations after graph is
                        quantized.


Example command for creating a staticly quantized mobilenet model:
python scripts/model_quantize_post_training.py \
    --onnx-file models/mobilenet_v1/model.onnx \
    --data-glob datasets/imagenet/val_samples \
    --output-model-path models/mobilenet_v1/model-quant.onnx
"""

import argparse

from neuralmagicML.onnx.utils import DataLoader
from neuralmagicML.onnx.quantization import quantize_model_post_training


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate and Quantize an Onnx model")
    parser.add_argument(
        "--onnx-file",
        required=True,
        type=str,
        help="File path to onnx model to quantize",
    )
    parser.add_argument(
        "--data-glob",
        required=True,
        type=str,
        help="Glob pattern to grab sample data files to feed through the "
        "model, sample data must be numpy files with one model input per file "
        "as either an array in an npy file or dictionary in an npz file, "
        "defaults to random",
    )
    parser.add_argument(
        "--output-model-path",
        required=True,
        type=str,
        help="File path to where the quantized model should be written",
    )
    parser.add_argument(
        "--op-types",
        type=str,
        default="Conv,MatMul,Gemm",
        help="Comma delimited operator types to be calibrated and quantized",
    )
    parser.add_argument(
        "--exclude-nodes",
        type=str,
        default="",
        help="Comma delimited operator names that should not be quantized",
    )
    parser.add_argument(
        "--include-nodes",
        type=str,
        default="",
        help="Comma delimited operator names force to be quantized",
    )
    parser.add_argument(
        "--augmented-model-path",
        type=str,
        default=None,
        help="Save augmented model to this file for verification purpose",
    )
    parser.add_argument(
        "--static",
        type=bool,
        default=True,
        help="Use static quantization. (Default is static)",
    )
    parser.add_argument(
        "--symmetric-weight",
        default=False,
        action="store_true",
        help="Use symmetric weight quantization. Default is False",
    )
    parser.add_argument(
        "--force-fusions",
        default=False,
        action="store_true",
        help="Set to force fusions in quantization. Default is False",
    )
    parser.add_argument(
        "--skip-extra-optimization",
        default=False,
        action="store_true",
        help="Set to not run extra optimizations after graph is quantized.",
    )

    args = parser.parse_args()
    args.op_types = args.op_types.split(",")
    args.exclude_nodes = args.exclude_nodes.split(",")
    args.include_nodes = args.include_nodes.split(",")

    return args


def main(args):
    data = DataLoader(data=args.data_glob, labels=None, batch_size=1)
    quantize_model_post_training(
        onnx_file=args.onnx_file,
        data_loader=data,
        output_model_path=args.output_model_path,
        calibrate_op_types=args.op_types,
        exclude_nodes=args.exclude_nodes,
        include_nodes=args.include_nodes,
        augmented_model_path=args.augmented_model_path,
        static=args.static,
        symmetric_weight=args.symmetric_weight,
        force_fusions=args.force_fusions,
        run_extra_opt=not args.skip_extra_optimization,
    )
    print("Quantized model saved to {}".format(args.output_model_path))


if __name__ == "__main__":
    args_ = parse_args()
    main(args_)
