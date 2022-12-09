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

from typing import List, Set, Union

from onnx import NodeProto


__all__ = ["assert_node_type"]


def assert_node_type(node: NodeProto, op: Union[List[str], Set[str], str]) -> bool:
    """
    Checks if a node is of the given op type
    :param node: the node to check
    :param op: the operation type to check for
    :return: True if the node has the given op type, False otherwise
    """
    if node is None:
        return False
    if isinstance(op, str):
        return node.op_type == op
    else:
        return node.op_type in op
