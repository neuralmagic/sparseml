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
Export TensorFlow models to the local device
"""

import os
from collections import OrderedDict
from typing import Dict, List, Union

import numpy
import onnx

from sparseml.tensorflow_v1.utils.helpers import tf_compat
from sparseml.tensorflow_v1.utils.variable import clean_tensor_name
from sparseml.utils import (
    clean_path,
    create_dirs,
    create_parent_dirs,
    path_file_count,
    tensors_export,
)


__all__ = ["default_onnx_opset", "GraphExporter"]


def default_onnx_opset() -> int:
    return 9 if onnx.__version__ < "1.6" else 11


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
        return os.path.join(self._output_dir, "tensorflow_v1")

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

    @staticmethod
    def pb_to_onnx(
        inputs: List[Union[str, tf_compat.Tensor]],
        outputs: List[Union[str, tf_compat.Tensor]],
        pb_path: str,
        onnx_path: str,
        opset: int = default_onnx_opset(),
        custom_op_handlers=None,
        extra_opset=None,
        shape_override: Dict[str, List] = None,
    ):
        """
        Export an ONNX format for the graph from PB format.
        Should not be called within an active graph or session.

        :param inputs: the inputs the graph should be created for,
            can be either a list of names or a list of tensors
        :param outputs: the outputs the graph should be created for,
            can be either a list of names or a list of tensors
        :param pb_path: path to the existing PB file
        :param onnx_path: path to the output ONNX file
        :param opset: ONNX opset
        :param custom_op_handlers: dictionary of custom op handlers
        :param extra_opset: list of extra opset's
        :param shape_override: new shape to override
        """
        try:
            from tf2onnx import constants, optimizer, utils
            from tf2onnx.tfonnx import process_tf_graph, tf_optimize
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "tf2onnx must be installed on the system before using export_onnx"
            )

        try:
            from tf2onnx import tf_loader as loader
        except Exception:
            from tf2onnx import loader

        pb_path = clean_path(pb_path)

        if not os.path.exists(pb_path):
            raise FileNotFoundError(
                ("no pb file for the model found at {}").format(pb_path)
            )

        inputs = [inp if isinstance(inp, str) else inp.name for inp in inputs]
        outputs = [out if isinstance(out, str) else out.name for out in outputs]

        graph_def, inputs, outputs = loader.from_graphdef(pb_path, inputs, outputs)
        graph_def = tf_optimize(inputs, outputs, graph_def, fold_constant=True)

        with tf_compat.Graph().as_default() as tf_graph:
            tf_compat.import_graph_def(graph_def, name="")

        with tf_compat.Session(graph=tf_graph):
            graph = process_tf_graph(
                tf_graph,
                continue_on_error=False,
                target=",".join(constants.DEFAULT_TARGET),
                opset=opset,
                custom_op_handlers=custom_op_handlers,
                extra_opset=extra_opset,
                shape_override=shape_override,
                input_names=inputs,
                output_names=outputs,
            )

        onnx_graph = optimizer.optimize_graph(graph)
        model_proto = onnx_graph.make_model("converted from {}".format(pb_path))

        onnx_path = clean_path(onnx_path)
        create_parent_dirs(onnx_path)
        utils.save_protobuf(onnx_path, model_proto)

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
        opset: int = default_onnx_opset(),
        custom_op_handlers=None,
        extra_opset=None,
        shape_override: Dict[str, List] = None,
    ):
        """
        Export an ONNX format for the graph from the PB format.
        Should not be called within an active graph or session.

        :param inputs: the inputs the graph should be created for,
            can be either a list of names or a list of tensors
        :param outputs: the outputs the graph should be created for,
            can be either a list of names or a list of tensors
        :param opset: ONNX opset
        :param custom_op_handlers: dictionary of custom op handlers
        :param extra_opset: list of extra opset's
        :param shape_override: new shape to override
        """
        GraphExporter.pb_to_onnx(
            inputs,
            outputs,
            self.pb_path,
            self.onnx_path,
            opset=opset,
            custom_op_handlers=custom_op_handlers,
            extra_opset=extra_opset,
            shape_override=shape_override,
        )

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
