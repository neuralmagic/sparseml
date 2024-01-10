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

try:
    from torch.distributed.fsdp import FullyShardedDataParallel
except ImportError:
    FullyShardedDataParallel = None

from contextlib import nullcontext


__all__ = ["summon_full_params_context", "fix_fsdp_module_name"]

FSDP_WRAPPER_NAME = "_fsdp_wrapped_module."


def summon_full_params_context(model):
    if FullyShardedDataParallel is not None:
        return FullyShardedDataParallel.summon_full_params(model)

    return nullcontext()


def fix_fsdp_module_name(name: str) -> str:
    """
    Remove FSDP wrapper prefixes from a module name

    :param name: name to strip
    :return: stripped name
    """
    return name.replace(FSDP_WRAPPER_NAME, "")
