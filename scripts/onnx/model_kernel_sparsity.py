"""
Script to analyze an ONNX model to get the kernel sparsity information
(results from pruning) for all prunable layers in the model.

##########
Command help:
usage: model_kernel_sparsity.py [-h] --onnx-file-path ONNX_FILE_PATH

Analyze an ONNX model to retrieve the kernel sparsity information for all
nodes in the model.

optional arguments:
  -h, --help            show this help message and exit
  --onnx-file-path ONNX_FILE_PATH
                        Path to the local onnx file to analyze


##########
Example:
python scripts/onnx/model_kernel_sparsity.py --onnx-file-path /PATH/TO/MODEL.onnx
"""

import argparse

from neuralmagicML.onnx.utils import onnx_nodes_sparsities


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze an ONNX model to retrieve the kernel sparsity "
        "information for all nodes in the model."
    )
    parser.add_argument(
        "--onnx-file-path",
        type=str,
        required=True,
        help="Path to the local onnx file to analyze",
    )

    return parser.parse_args()


def main(args):
    print("analyzing model")
    total_sparse, node_sparse = onnx_nodes_sparsities(args.onnx_file_path)

    print("node inp sparsities:")
    for name, val in node_sparse.items():
        print("{}: {}".format(name, val))

    print("\ntotal sparsity: {}".format(total_sparse))


if __name__ == "__main__":
    args_ = parse_args()
    main(args_)
