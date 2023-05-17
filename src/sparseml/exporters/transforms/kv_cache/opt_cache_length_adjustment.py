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


import logging
from typing import List, Tuple

import numpy
import onnx
from onnx import ModelProto, NodeProto, numpy_helper

from sparseml.exporters.transforms.kv_cache.cache_length_adjustment import (
    CacheLengthAdjustment,
)
from sparseml.onnx.utils import ONNXGraph


__all__ = ["OPTCacheLengthAdjustment"]


_LOGGER = logging.getLogger(__name__)


# name of model weight for the embed positions
# accessing these must be updated according to the cache length
_EMBED_POSITIONS_ID = "model.decoder.embed_positions.weight"


class OPTCacheLengthAdjustment(CacheLengthAdjustment):
    """
    Updates OPT model masking to account for the length of the KV cache

    Specifically, slices the embedding position lookup based on cache length

    Transforms
    ```
    |  Initial gather targets
    |     |
    |  Gather(model.decoder.embed_positions.weight)
    ```
    Into
    ```
    |  Initial gather targets               Unsqueeze(CACHE_LENGTH)
    |     |                                         |
    |                           Slice
    |     |
    |  Gather(model.decoder.embed_positions.weight)
    ```
    """

    def update_model_for_cache_length(self, model: ModelProto) -> ModelProto:
        """
        updates the model to handle cache length after a cache_length
        graph input has been added

        :param model: model to update
        :return: updated model
        """
        self._cache_length_unsqueeze_name = "cache_length_unsqueezed"
        self._slice_minus_two_name = "slice_minus_2_constant"
        self._slice_zero_name = "slice_0_constant"
        self._slice_steps_one_name = "slice_steps_1_constant"

        model = self._update_position_embeddings_gather(model)

        model = self._add_slice_constants(model)

        for concat_node, cache_idx in _find_cache_concat_node_with_idx(model):
            model = self._update_cache_concat_for_cache_length(
                concat_node, cache_idx, model
            )

        return model

    def _add_slice_constants(self, model: ModelProto) -> ModelProto:
        slice_minus_two_init = numpy_helper.from_array(
            numpy.array([-2], dtype=numpy.int64),
            self._slice_minus_two_name,
        )
        slice_zero_init = numpy_helper.from_array(
            numpy.array([0], dtype=numpy.int64),
            self._slice_zero_name,
        )
        slice_steps_initializer = numpy_helper.from_array(
            numpy.ones(1, dtype=numpy.int64),
            self._slice_steps_one_name,
        )
        model.graph.initializer.append(slice_minus_two_init)
        model.graph.initializer.append(slice_zero_init)
        model.graph.initializer.append(slice_steps_initializer)
        return model

    def _update_cache_concat_for_cache_length(
        self, concat_node: NodeProto, cache_idx: int, model: ModelProto
    ) -> ModelProto:

        cache_name = concat_node.input[cache_idx]

        slice_name = f"{cache_name}.slice"
        slice_node = onnx.helper.make_node(
            op_type="Slice",
            inputs=[
                cache_name,  # rewire gather input to slice
                self._slice_zero_name,  # start from 0
                self._cache_length_unsqueeze_name,  # end at cache length
                self._slice_minus_two_name,
                self._slice_steps_one_name,
            ],
            outputs=[slice_name],
            name=slice_name,
        )

        # rewire concat input to be slice output
        concat_node.input[cache_idx] = slice_node.output[0]

        # insert slice node before the concat node
        concat_node_model_idx = [
            idx
            for idx, node in enumerate(model.graph.node)
            if node.name == concat_node.name
        ][0]
        model.graph.node.insert(concat_node_model_idx, slice_node)

        return model

    def _update_position_embeddings_gather(self, model: ModelProto) -> ModelProto:
        # get gather node, raise if cannot find
        embed_positions_gather_node = _find_embed_positions_gather_node(model)

        # create initializer helpers for slice
        slice_node_name = "embed_positions_slice"
        slice_ends_initializer = numpy_helper.from_array(
            numpy.array([numpy.iinfo(numpy.int64).max]),  # do not cap end of slice
            f"{slice_node_name}.ends",
        )
        slice_axes_initializer = numpy_helper.from_array(
            numpy.ones(1, dtype=numpy.int64),
            f"{slice_node_name}.axes",
        )
        axes = numpy_helper.from_array(
            numpy.array([0], dtype=numpy.int64),
            f"axes",
        )

        # unsqueeze dim 0 of cache length to align for slicing the right dim
        unsqueeze_output_name = "cache_length_unsqueezed"
        unsqueeze_node = onnx.helper.make_node(
            op_type="Unsqueeze",
            inputs=[self.CACHE_LENGTH_NAME, axes.name],
            outputs=[self._cache_length_unsqueeze_name],
            name=self._cache_length_unsqueeze_name,
        )
        # create slice node to select only from cache length
        slice_node = onnx.helper.make_node(
            op_type="Slice",
            inputs=[
                embed_positions_gather_node.input[1],  # rewire gather input to slice
                unsqueeze_output_name,  # start from cache length (unsqueezed)
                slice_ends_initializer.name,
                slice_axes_initializer.name,
                self._slice_steps_one_name,
            ],
            outputs=[slice_node_name],
            name=slice_node_name,
        )

        # rewire gather to read from slice
        embed_positions_gather_node.input[1] = slice_node.output[0]

        # add nodes and initializers to model
        # model.graph.node.extend([unsqueeze_node, slice_node])
        model.graph.initializer.extend(
            [
                slice_ends_initializer,
                slice_axes_initializer,
                axes,
            ]
        )
        model.graph.node.insert(
            [
                i
                for i, n in enumerate(model.graph.node)
                if n.name == embed_positions_gather_node.name
            ][0],
            slice_node,
        )

        model.graph.node.insert(
            [i for i, n in enumerate(model.graph.node) if n.name == slice_node.name][0],
            unsqueeze_node,
        )

        return model


def _find_embed_positions_gather_node(model: ModelProto) -> NodeProto:
    for node in model.graph.node:
        if node.op_type != "Gather":
            continue
        if node.input[0] == _EMBED_POSITIONS_ID:
            # found the embed_positions_gather_node
            return node
    raise ValueError(
        f"Unable to find embed positions gather node with id {_EMBED_POSITIONS_ID} "
        "in OPT cache length adjustment"
    )


def _find_cache_concat_node_with_idx(model: ModelProto) -> List[Tuple[NodeProto, int]]:
    # return tuples of (concat_node, cache_input_idx)
    concat_nodes_and_idxs = []

    for node in model.graph.node:
        if node.op_type != "Concat":
            continue
        for idx, input_name in enumerate(node.input):
            if "past_key_values" in input_name:
                concat_nodes_and_idxs.append((node, idx))
                break  # break inner loop
    return concat_nodes_and_idxs
