"""
Ease of use script for converting an ONNX model exported from PyTorch quantization
aware training to a quantized ONNX model.  All QAT quantized ops must be surrounded
by a quantization observer in torch (exported as a QuantizeLinear op immediately
followed by a DequantizeLinear op).

##########
Command help:
usage: model_quantize_qat_export.py [-h] --onnx-file-path ONNX_FILE_PATH
                                    [--output-file-path OUTPUT_FILE_PATH]

convert an ONNX model exported from PyTorch quantization aware training to a
quantized ONNX model

optional arguments:
  -h, --help            show this help message and exit
  --onnx-file-path ONNX_FILE_PATH
                        Path to the local onnx file to convert
  --output-file-path OUTPUT_FILE_PATH
                        Path to save the converted model to, defaults to the
                        same file path as the input file with '-quantized'
                        appended to the file name
##########
Example:
python3 scripts/python/model_quantize_qat_export.py \
  --onnx-file-path /PATH/TO/QAT-MODEL.onnx

"""


import argparse
import onnx

from neuralmagicML import get_main_logger
from neuralmagicML.pytorch.quantization import quantize_torch_qat_export


LOGGER = get_main_logger()


def parse_args():
    parser = argparse.ArgumentParser(
        description="convert an ONNX model exported from PyTorch quantization aware"
        " training to a quantized ONNX model"
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
        help="Path to save the converted model to, defaults to the same file path"
        " as the input file with '-quantized' appended to the file name",
    )

    return parser.parse_args()


def main(args):
    model = onnx.load(args.onnx_file_path)

    LOGGER.info("converting model...")
    quantize_torch_qat_export(model)

    output_file_path = (
        args.output_file_path
        if args.output_file_path
        else "{}-quantized.onnx".format(args.onnx_file_path.split(".onnx")[0])
    )
    onnx.save(model, output_file_path)
    LOGGER.info("saved converted model to {}".format(output_file_path))


if __name__ == "__main__":
    args_ = parse_args()
    main(args_)
