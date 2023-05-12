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
        # get gather node, raise if cannot find
        embed_positions_gather_node = _find_embed_positions_gather_node(model)

        # create initializer helpers for slice
        slice_node_name = "embed_positions_slice"
        slice_ends_initializer = numpy_helper.from_array(
            numpy.array(numpy.iinfo(numpy.int64).max),  # do not cap end of slice
            f"{slice_node_name}.ends",
        )
        slice_axes_initializer = numpy_helper.from_array(
            numpy.ones(1, dtype=numpy.int64),
            f"{slice_node_name}.axes",
        )
        slice_steps_initializer = numpy_helper.from_array(
            numpy.ones(1, dtype=numpy.int64),
            f"{slice_node_name}.stpes",
        )

        # unsqueeze dim 0 of cache length to align for slicing the right dim
        unsqueeze_ouptut_name = "cache_length_unsqueezed"
        unsqueeze_node = onnx.helper.make_node(
            op_type="Unsqueeze",
            inputs=[self.CACHE_LENGTH_NAME],
            outputs=[unsqueeze_ouptut_name],
            axes=0,
            name=unsqueeze_ouptut_name,
        )
        # create slice node to select only from cache length
        slice_node = onnx.helper.make_node(
            op_type="Slice",
            inputs=[
                embed_positions_gather_node.input[1],  # rewire gather input to slice
                unsqueeze_ouptut_name,  # start from cache length (unsqueezed)
                slice_ends_initializer.name,
                slice_axes_initializer.name,
                slice_steps_initializer.name,
            ],
            outputs=[slice_node_name],
            name=slice_node_name,
        )

        # rewire gather to read from slice
        embed_positions_gather_node.input[1] = slice_node.output[0]

        # add nodes and initializers to model
        model.graph.node.extend([unsqueeze_node, slice_node])
        model.graph.initializer.extend(
            [slice_ends_initializer, slice_axes_initializer, slice_steps_initializer]
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
