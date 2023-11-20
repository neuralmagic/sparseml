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


import pytest

from sparseml.core.framework import Framework
from sparseml.core.model.base import ModifiableModel
from sparseml.core.state import State
from tests.sparseml.helpers import should_skip_pytorch_tests


def get_linear_net_with_device(device="cpu"):
    from tests.sparseml.pytorch.helpers import LinearNet

    class LinearNetWithMockDevice(LinearNet):
        def __init__(self):
            super().__init__()
            self.device = device

        def to(self, device):
            # Do not need to actually move
            # the model to the device

            # uncomment next line to actually move
            # super().to(device)
            self.device = device
            return self

    return LinearNetWithMockDevice()


class TestState:
    @pytest.mark.parametrize(
        "kwargs, expected",
        [
            ({"framework": Framework.pytorch}, False),
            ({"framework": Framework.pytorch, "model": 1}, False),
            ({"framework": Framework.pytorch, "model": 1, "optimizer": 1}, True),
        ],
    )
    def test_sparsification_ready(self, kwargs, expected):
        state = State(**kwargs)
        assert state.sparsification_ready == expected

    @pytest.mark.parametrize("start", [1, None])
    def test_update_with_start_sets_start_event(self, start):
        state = State(framework=Framework.pytorch)
        state.update(start=start)
        if start is not None:
            assert state.start_event is not None
            assert state.start_event.current_index == start
        else:
            assert state.start_event is None

    @pytest.mark.parametrize(
        "arg_name, arg_value",
        [
            ("batches_per_step", 1),
            ("batches_per_step", None),
            ("steps_per_epoch", 1),
            ("steps_per_epoch", None),
        ],
    )
    def test_update_sets_start_event(self, arg_name, arg_value):
        state = State(framework=Framework.pytorch)
        state.update(**{arg_name: arg_value})
        if arg_value is not None:
            assert state.start_event is not None
            assert getattr(state.start_event, arg_name) == arg_value
        else:
            assert state.start_event is None

    @pytest.mark.parametrize(
        "data_arg, data_value",
        [
            ("train_data", 1),
            ("test_data", 1),
            ("val_data", 1),
            ("calib_data", 1),
        ],
    )
    def test_update_sets_data(self, data_arg, data_value):
        state = State(framework=Framework.pytorch)
        state.update(**{data_arg: data_value})

        # remove _data suffix
        data_arg_key = data_arg[:-5]
        if data_value is not None:
            assert getattr(state.data, data_arg_key) == data_value

    def test_update_can_set_teacher_model(self):
        state = State(framework=Framework.pytorch)
        state.update(teacher_model=1)
        assert state.teacher_model is not None
        assert isinstance(state.teacher_model, ModifiableModel)
        assert state.teacher_model.model == 1

    @pytest.mark.skipif(
        should_skip_pytorch_tests(),
        reason="Skipping pytorch tests either torch is not installed or "
        "NM_ML_SKIP_PYTORCH_TESTS is set",
    )
    def test_update_auto_moves_model_to_device(self):
        state = State(framework=Framework.pytorch)
        model = get_linear_net_with_device(device="cuda")
        assert model.device == "cuda"
        assert state.hardware.device is None

        state.update(model=model, device="cpu")
        assert state.model.model == model

        # update does not move the model to the device on initialization, it is a
        # parameter used by individual modifiers to control device management
        assert state.model.model.device == "cuda"
        assert state.hardware.device == "cpu"
