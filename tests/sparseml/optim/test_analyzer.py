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

import os
import tempfile

from sparseml.optim import AnalyzedLayerDesc


def test_layer_descs():
    descs = [
        AnalyzedLayerDesc("layer1", "Linear"),
        AnalyzedLayerDesc("layer2", "Conv2d"),
    ]
    tempdir = tempfile.gettempdir()
    save_path = os.path.join(tempdir, "layer_descs.json")

    AnalyzedLayerDesc.save_descs(descs, save_path)
    loaded_descs = AnalyzedLayerDesc.load_descs(save_path)

    for desc, loaded_desc in zip(descs, loaded_descs):
        assert desc.name == loaded_desc.name
        assert desc.type_ == loaded_desc.type_
    os.remove(save_path)
