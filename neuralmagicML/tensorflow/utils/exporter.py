"""
Export TensorFlow models to the local device
"""

from typing import List, Dict, Union
import os
from collections import OrderedDict
import numpy

from neuralmagicML.utils import (
    clean_path,
    create_parent_dirs,
    create_dirs,
    path_file_count,
)
from neuralmagicML.tensorflow.utils.helpers import tf_compat, tensors_export
from neuralmagicML.tensorflow.utils.variable import clean_tensor_name


__all__ = ["GraphExporter"]


class GraphExporter(object):
    """
    An exporter for exporting TensorFlow graphs and sessions into ONNX format
    as well as numpy arrays for the intput and output tensors

    :param output_dir: the directory to save the exports to
    """

    def __init__(self, output_dir: str):
        self._output_dir = clean_path(output_dir)

    @property
    def tensorflow_path(self) -> str:
        return os.path.join(self._output_dir, "tensorflow")

    @property
    def checkpoint_path(self) -> str:
        return os.path.join(self.tensorflow_path, "model")

    @property
    def pb_path(self) -> str:
        return os.path.join(self.tensorflow_path, "model.pb")

    @property
    def onnx_path(self) -> str:
        return os.path.join(self._output_dir, "model.onnx")

    @property
    def sample_inputs_path(self) -> str:
        return os.path.join(self._output_dir, "_sample-inputs")

    @property
    def sample_outputs_path(self) -> str:
        return os.path.join(self._output_dir, "_sample-outputs")

    def export_checkpoint(
        self, saver: tf_compat.train.Saver = None, sess: tf_compat.Session = None
    ):
        """
        Export a checkpoint for the current TensorFlow graph and session.

        :param saver: the saver instance to use to save the current session,
            if not supplied will create a new one using TRAINABLE_VARIABLES
        :param sess: the current session to export a checkpoint for,
            if not supplied will use get_default_session()
        """
        if not sess:
            sess = tf_compat.get_default_session()

        if not saver:
            saver = tf_compat.train.Saver(
                tf_compat.get_collection(tf_compat.GraphKeys.TRAINABLE_VARIABLES)
            )

        create_parent_dirs(self.checkpoint_path)
        saver.save(sess, self.checkpoint_path)

    def export_pb(
        self,
        outputs: List[Union[str, tf_compat.Tensor]],
        graph: tf_compat.Graph = None,
        sess: tf_compat.Session = None,
    ):
        """
        Export a serialized pb version of the a graph and session.

        :param outputs: the list of outputs the graph should be created for
            (used to determine the scope of the graph to export),
            can be either a list of names or a list of tensors
        :param graph: the graph to export to a pb format,
            if not supplied will use get_default_graph()
        :param sess: the session to export to a pb format,
            if not supplied will use get_default_session()
        """
        if not graph:
            graph = tf_compat.get_default_graph()

        if not sess:
            sess = tf_compat.get_default_session()

        outputs = [
            out if isinstance(out, str) else clean_tensor_name(out) for out in outputs
        ]
        output_graph_def = tf_compat.graph_util.convert_variables_to_constants(
            sess, graph.as_graph_def(), outputs
        )
        create_parent_dirs(self.pb_path)

        with tf_compat.gfile.GFile(self.pb_path, "wb") as f:
            f.write(output_graph_def.SerializeToString())

    def export_onnx(
        self,
        inputs: List[Union[str, tf_compat.Tensor]],
        outputs: List[Union[str, tf_compat.Tensor]],
    ):
        """
        Export an ONNX format for the graph

        :param inputs: the inputs the graph should be created for,
            can be either a list of names or a list of tensors
        :param outputs: the outputs the graph should be created for,
            can be either a list of names or a list of tensors
        """
        inputs = [inp if isinstance(inp, str) else inp.name for inp in inputs]
        outputs = [out if isinstance(out, str) else out.name for out in outputs]

        try:
            from tf2onnx.tfonnx import process_tf_graph, tf_optimize
            from tf2onnx import constants, loader, utils, optimizer
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "tf2onnx must be installed on the system before using export_onnx"
            )

        if not os.path.exists(self.pb_path):
            raise FileNotFoundError(
                (
                    "export_pb must be called first, "
                    "no pb file for the model found at {}"
                ).format(self.pb_path)
            )

        graph_def, inputs, outputs = loader.from_graphdef(self.pb_path, inputs, outputs)
        graph_def = tf_optimize(inputs, outputs, graph_def, True)

        with tf_compat.Graph().as_default() as tf_graph:
            tf_compat.import_graph_def(graph_def, name="")

        with tf_compat.Session(graph=tf_graph):
            graph = process_tf_graph(
                tf_graph,
                continue_on_error=False,
                target=",".join(constants.DEFAULT_TARGET),
                opset=11,
                custom_op_handlers={},
                extra_opset=[],
                shape_override=None,
                input_names=inputs,
                output_names=outputs,
            )

        onnx_graph = optimizer.optimize_graph(graph)
        model_proto = onnx_graph.make_model("converted from {}".format(self.pb_path))

        create_parent_dirs(self.onnx_path)
        utils.save_protobuf(self.onnx_path, model_proto)

    def export_samples(
        self,
        inp_tensors: List[tf_compat.Tensor],
        inp_vals: List[numpy.ndarray],
        out_tensors: List[tf_compat.Tensor],
        sess: tf_compat.Session,
    ) -> List[tf_compat.Tensor]:
        """
        Export sample tensors for the model to the local system.
        Executes the inputs through the model using a session to get the outputs.

        :param inp_tensors: the input tensors to feed through the model
        :param inp_vals: the input values to feed through the model and save
        :param out_tensors: the output tensors to load values from the model
            for saving
        :param sess: the session to export to a pb format,
            if not supplied will use get_default_session()
        """
        if not sess:
            sess = tf_compat.get_default_session()

        inp_dict = OrderedDict(
            [(tens, val) for tens, val in zip(inp_tensors, inp_vals)]
        )
        out_vals = sess.run(out_tensors, feed_dict=inp_dict)
        out_dict = OrderedDict(
            [(tens, val) for tens, val in zip(out_tensors, out_vals)]
        )
        self.export_named_samples(inp_dict, out_dict)

        return out_vals

    def export_named_samples(
        self,
        inp_dict: Dict[Union[str, tf_compat.Tensor], numpy.ndarray],
        out_dict: Dict[Union[str, tf_compat.Tensor], numpy.ndarray],
    ):
        """
        Export sample inputs and outputs for the model to the local system.

        :param inp_dict: the inputs to save
        :param out_dict: the outputs to save
        """
        inp_dict = OrderedDict(
            [
                (tens if isinstance(tens, str) else tens.name, val)
                for tens, val in inp_dict.items()
            ]
        )
        out_dict = OrderedDict(
            [
                (tens if isinstance(tens, str) else tens.name, val)
                for tens, val in out_dict.items()
            ]
        )
        create_dirs(self.sample_inputs_path)
        create_dirs(self.sample_outputs_path)
        exp_counter = path_file_count(self.sample_inputs_path, "inp*.npz")
        tensors_export(
            inp_dict,
            self.sample_inputs_path,
            name_prefix="inp",
            counter=exp_counter,
            break_batch=True,
        )
        tensors_export(
            out_dict,
            self.sample_outputs_path,
            name_prefix="out",
            counter=exp_counter,
            break_batch=True,
        )
