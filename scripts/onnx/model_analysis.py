"""
Script to analyze an ONNX model to get all of the nodes in the model
and parse relevant information for them.

Includes whether nodes are prunable, param counts, flops (TODO), etc.


##########
Command help:
usage: model_analysis.py [-h] --onnx-file-path ONNX_FILE_PATH
                         [--output-path OUTPUT_PATH]

Analyze an ONNX model to parse it into relevant information such as prunable
nodes, params count, flops (TODO), etc

optional arguments:
  -h, --help            show this help message and exit
  --onnx-file-path ONNX_FILE_PATH
                        Path to the local onnx file to analyze
  --output-path OUTPUT_PATH
                        Path to save the output json file to, defaults to
                        save next to the onnx-file-path


##########
Example:
python scripts/onnx/model_analysis.py --onnx-file-path /PATH/TO/MODEL.onnx
"""

import argparse

from neuralmagicML.onnx.recal import ModelAnalyzer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze an ONNX model to parse it into relevant information "
        "such as prunable nodes, params count, flops (TODO), etc"
    )
    parser.add_argument(
        "--onnx-file-path",
        type=str,
        required=True,
        help="Path to the local onnx file to analyze",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path to save the output json file to, "
        "defaults to save next to the onnx-file-path",
    )

    return parser.parse_args()


def main(args):
    print("analyzing model")
    analyzer = ModelAnalyzer(args.onnx_file_path)

    save_path = (
        args.output_path
        if args.output_path is not None
        else "{}.analyzed.json".format(args.onnx_file_path)
    )
    print("analyzed, saving to {}".format(save_path))

    analyzer.save_json(save_path)


if __name__ == "__main__":
    args_ = parse_args()
    main(args_)
