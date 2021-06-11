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
Implementations for an Identity module.
"""

from typing import Callable, Optional

from torch.nn import Module


class Identity(Module):
    """
    A placeholder identity operator that is argument-insensitive.

    :param mapping_lambda: A lambda callable to map *args and **kwargs from the
        forward function to a return value.
        Defaults to returning all positional arguments as a Tuple.
        If only one argument returned, returns individual.
    """

    def __init__(self, mapping_lambda: Optional[Callable] = None, *args, **kwargs):
        super(Identity, self).__init__()
        self._mapping_lambda = mapping_lambda or self._standard_forward

    def forward(self, *args, **kwargs):
        return self._mapping_lambda(*args, **kwargs)

    def _standard_forward(self, *args, **kwargs):
        # ignore the keyword args and return only args
        # if args is one, then return as a single item for usual tensors
        # if args is multiple, return as Tuple
        args = tuple(args)

        return args[0] if len(args) == 1 else args
