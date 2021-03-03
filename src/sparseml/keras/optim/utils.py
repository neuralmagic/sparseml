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


__all__ = ["get_layer_name_from_param"]


def get_layer_name_from_param(param: str):
    known_weights = ["kernel", "bias"]
    pos = param.rfind("/")
    if pos > -1:
        suff = param[pos + 1 :]
        found = False
        for s in known_weights:
            colon_pos = suff.rfind(":")
            if suff[:colon_pos] == s:
                found = True
                break
        if not found:
            raise ValueError(f"Unrecognized weight names. Expected: {known_weights}")
    return param[:pos]
