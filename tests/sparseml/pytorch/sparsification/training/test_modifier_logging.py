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

import pytest

from sparseml.pytorch.sparsification.training import SparsificationLoggingModifier


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "start_epoch, end_epoch, update_frequency",
    [
        (0.0, 10.0, 2),
        (5.0, -1, 1),
        (0.0, 1.0, -1),
    ],
)
def test_epoch_range_yaml(start_epoch, end_epoch, update_frequency):
    yaml_str = """
    !SparsificationLoggingModifier
        start_epoch: {start_epoch}
        end_epoch: {end_epoch}
        update_frequency: {update_frequency}
    """.format(
        start_epoch=start_epoch, end_epoch=end_epoch, update_frequency=update_frequency
    )
    yaml_modifier = SparsificationLoggingModifier.load_obj(yaml_str)
    serialized_modifier = SparsificationLoggingModifier.load_obj(str(yaml_modifier))
    obj_modifier = SparsificationLoggingModifier(
        start_epoch=start_epoch, end_epoch=end_epoch, update_frequency=update_frequency
    )

    assert isinstance(yaml_modifier, SparsificationLoggingModifier)
    assert (
        yaml_modifier.start_epoch
        == serialized_modifier.start_epoch
        == obj_modifier.start_epoch
    )
    assert (
        yaml_modifier.end_epoch
        == serialized_modifier.end_epoch
        == obj_modifier.end_epoch
    )
    assert (
        yaml_modifier.update_frequency
        == serialized_modifier.update_frequency
        == obj_modifier.update_frequency
    )
