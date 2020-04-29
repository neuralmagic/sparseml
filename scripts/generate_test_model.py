"""
Generate a tensorflow test model using a random model from TF's nets_factory

============================================================================

For help, run python3 scripts/generate_test_model.py --help

Installation TF slim's models at:
https://github.com/tensorflow/models/tree/master/research/slim#installing-the-tf-slim-image-models-library

Example usage:
    python3 scripts/generate_test_model.py \
    --model_name vgg_16 \
    --inputs input:0 \
    --outputs vgg_16/fc8/squeezed:0 \
    --output_dir cv-classification/vgg/16-randomized-tensorflow-dense
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os
from typing import Dict, List

import numpy as np
import onnxruntime as rt
from tensorflow.contrib import slim

from neuralmagicML.tensorflow.utils import GraphExporter, nets_utils, tf_compat as tf


_HELP_TEXT = """
Loads a tensorflow model from tensorflow's slim/nets/nets_factory.
Saves a frozen model proto, tf2onnx conversion, and sample inputs and outputs.
"""


def get_args_and_exporter():
    """Parse commandline."""
    parser = argparse.ArgumentParser(
        description="Generate model proto, sample inputs, outputs, and onnx files for"
        " tensorflow models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_HELP_TEXT,
    )
    parser.add_argument(
        "--model_name",
        help="Name of the model to be taken from tensorflow/slim nets_factory",
    )
    parser.add_argument("--inputs", help="input node names, comma separated")
    parser.add_argument("--outputs", help="output node names, comma separated")
    parser.add_argument("--output_dir", help="file output directory")
    parser.add_argument(
        "--checkpoint_file", help="[Optional] Path to .ckpt file for pretrained weights"
    )
    parser.add_argument(
        "--arg_scope_vars",
        help="[Optional] Comma separated list of variables to be passed to the models"
        " arg_scope function.  ex --arg_scope_vars weight_decay=.01,std=.2",
    )
    parser.add_argument(
        "--num_classes",
        help="number of model classes",
        type=int,
        default=1000
    )
    parser.add_argument(
        "--num_image_dims",
        help="Dimensions of input image, default=2 for a 2-d image.",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--save_old_sample_inputs_outputs",
        help="Add flag if old sample inputs and outputs should not be overwritten",
    )
    parser.add_argument(
        "--num_samples",
        help="The number of sample inputs and outputs to generate",
        type=int,
        default=100,
    )
    args = parser.parse_args()
    args.inputs = args.inputs.split(",")
    args.outputs = args.outputs.split(",")
    if args.arg_scope_vars:
        args.arg_scope_vars = [var.split("=") for var in args.arg_scope_vars.split(",")]
        args.arg_scope_vars = {var[0]: var[1] for var in args.arg_scope_vars}
    else:
        args.arg_scope_vars = {}

    # make model subdirectory if it does not exist and set output_dir to it
    output_sub_dirs = os.listdir(args.output_dir)
    args.output_dir = "{}/{}".format(args.output_dir, args.model_name)
    if args.model_name not in output_sub_dirs:
        os.mkdir(args.output_dir)

    file_subdirs = os.listdir(args.output_dir)
    for subdir in ["tensorflow", "_sample-inputs", "_sample-outputs"]:
        if subdir not in file_subdirs:
            os.mkdir("{}/{}".format(args.output_dir, subdir))
    exporter = GraphExporter(args.output_dir)

    def delete_old_files(dir_path):
        """
        deletes files in directory dir_path
        """
        old_files = os.listdir(dir_path)
        for file in old_files:
            os.remove(os.path.join(dir_path, file))

    if not args.save_old_sample_inputs_outputs:
        delete_old_files(exporter.sample_inputs_path)
        delete_old_files(exporter.sample_outputs_path)

    return args, exporter


def freeze_test_model(
    model_name: str,
    output_node_names: List[str],
    exporter: GraphExporter,
    num_classes: int = 1000,
    num_image_dims: int = 2,
    checkpoint_file: str = None,
    arg_scope_vars: Dict = {},
):
    # Load graph
    graph = tf.get_default_graph()
    network_fn = nets_utils.get_network_fn(
        model_name, num_classes, arg_scope_vars=arg_scope_vars
    )
    # Freeze graph
    image_dims_shape = [network_fn.default_image_size] * num_image_dims
    placeholder = tf.placeholder(
        "float", name="input", shape=(None, *image_dims_shape, 3)
    )
    _, end_points = network_fn(placeholder)
    graph_def = graph.as_graph_def()
    output_nodes = [name.split(":")[0] for name in output_node_names]

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    if checkpoint_file is not None:  # load from checkpoint file
        model_scope_name = output_node_names[0].split("/")[0]
        variables_to_restore = slim.get_model_variables(model_scope_name)
        # restorer.restore(sess, checkpoint_file)
        init_fn = slim.assign_from_checkpoint_fn(checkpoint_file, variables_to_restore)
        init_fn(sess)
    # write frozen graph pb and full checkpoint files
    exporter.export_pb(output_nodes, graph, sess)
    exporter.export_checkpoint(tf.train.Saver(max_to_keep=1), sess)
    sess.close()


def save_sample_inputs_and_outputs(
    input_node_names: List[str],
    output_node_names: List[str],
    exporter: GraphExporter,
    num_samples: int,
):
    # Load frozen graph
    with tf.gfile.GFile(exporter.pb_path, "rb") as f:
        restored_graph_def = tf.GraphDef()
        restored_graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(restored_graph_def)
    graph_def = graph.as_graph_def()
    # Define input and output tensors
    input_tensors = [
        graph.get_tensor_by_name("import/" + input_node_name)
        for input_node_name in input_node_names
    ]
    output_tensors = [
        graph.get_tensor_by_name("import/" + output_node_name)
        for output_node_name in output_node_names
    ]

    def get_sample_input_data(tensor):
        shape = [dim.value for dim in tensor.shape]
        shape[0] = 1  # set batch size to 1
        return np.random.randn(*shape).astype(np.float32) * np.sqrt(
            1 / (np.prod(shape) - 1)
        )

    for i in range(num_samples):
        # Generate sample inputs and outputs
        sample_inputs = [
            get_sample_input_data(input_tensor) for input_tensor in input_tensors
        ]
        # sample_feed_dict = dict(zip(input_tensors, sample_inputs))
        sess = tf.Session(graph=graph)
        exporter.export_samples(input_tensors, sample_inputs, output_tensors, sess)


def convert_to_onnx(
    input_node_names: List[str], output_node_names: List[str], exporter: GraphExporter,
):
    exporter.export_onnx(input_node_names, output_node_names)


def run_sample_onnxrumtime(exporter: GraphExporter, sample_number: int = 0):
    # Load onnx model and get input and output names
    sess = rt.InferenceSession(exporter.onnx_path)
    input_names = [input.name for input in sess.get_inputs()]
    output_names = [output.name for output in sess.get_outputs()]
    # Generate sample data
    sample_name = "inp-{:04}.npz".format(sample_number)
    sample_inputs = np.load(os.path.join(exporter.sample_inputs_path, sample_name))
    sample_inputs = [
        np.expand_dims(sample_input.astype(np.float32), 0)
        for sample_input in sample_inputs.values()
    ]
    inputs_dict = dict(zip(input_names, sample_inputs))
    # Run model
    onnx_outputs = sess.run(output_names, inputs_dict)
    return onnx_outputs


def compare_tf_onnx_outputs(
    onnx_outputs: List[np.ndarray], exporter: GraphExporter, sample_number: int = 0
):
    """
    Compares the output arrays from running the sample inputs on the onnx
    runtime versus tensorflow.  Raises an error if the maximum absolute error
    is greater than 1e-4
    """
    sample_name = "out-{:04}.npz".format(sample_number)
    sample_outputs = np.load(os.path.join(exporter.sample_outputs_path, sample_name))
    tf_outputs = [np.expand_dims(arr, 0) for arr in sample_outputs.values()]
    for onnx_output, tf_output in zip(onnx_outputs, tf_outputs):
        if np.max(np.abs(onnx_output - tf_output)) >= 1e-4:
            raise Exception(
                "Absolute maximum error between tf and onnx" " greater than 1e-4."
            )


def generate_test_model(
    model_name: str,
    exporter: GraphExporter,
    inputs: List[str],
    outputs: List[str],
    num_classes: int,
    num_image_dims: int,
    arg_scope_vars: Dict,
    checkpoint_file: str,
    num_samples: int,
):
    """
    Loads a freezes a TF model.  Saves frozen model with its checkpoint.
    Generates sample inputs and outputs for the model.
    Converts model to ONNX and tests that ONNX run time and TF have the same
    outputs for the sample inputs.

    :param model_name: name of the model as defined in TF slim's nets_factory
    :param exporter: GraphExporter object for the model
    :param inputs: list of tensor names that are inputs to the model
    :param outputs: list of tensor names that are outputs of the model
    :param num_classes: the number of output classes that the model should have
    :param num_image_dims: the number of dimensions in input images (ie 2 or 3)
    :param arg_scope_vars: dictionary of optional variables for nets_factory
    :param checkpoint_file: path to .ckpt file of model weights
    :param num_samples: number of sample inputs and outputs to generate
    """
    freeze_test_model(
        model_name,
        outputs,
        exporter,
        num_classes=num_classes,
        num_image_dims=num_image_dims,
        checkpoint_file=checkpoint_file,
        arg_scope_vars=arg_scope_vars,
    )
    tf.reset_default_graph()
    convert_to_onnx(inputs, outputs, exporter)
    tf.reset_default_graph()
    save_sample_inputs_and_outputs(inputs, outputs, exporter, num_samples)
    for i in range(num_samples):
        onnx_outputs = run_sample_onnxrumtime(exporter, sample_number=i)
        compare_tf_onnx_outputs(onnx_outputs, exporter, sample_number=i)


def main():
    args, exporter = get_args_and_exporter()
    generate_test_model(
        args.model_name,
        exporter,
        args.inputs,
        args.outputs,
        num_classes=args.num_classes,
        num_image_dims=args.num_image_dims,
        arg_scope_vars=args.arg_scope_vars,
        checkpoint_file=args.checkpoint_file,
        num_samples=args.num_samples,
    )


if __name__ == "__main__":
    main()
