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

from sparseml.optim import BaseManager, BaseModifier, BaseScheduled
from sparseml.sparsification.types import SparsificationTypes


class _Pruning(BaseScheduled, BaseModifier):
    @BaseModifier.sparsification_types.getter
    def sparsification_types(self):
        return [SparsificationTypes.pruning]


class _Quant(BaseScheduled, BaseModifier):
    @BaseModifier.sparsification_types.getter
    def sparsification_types(self):
        return [SparsificationTypes.quantization]


def test_manager():
    manager = BaseManager(
        modifiers=[
            BaseScheduled(
                start_epoch=1.0,
                min_start=0,
                end_epoch=2.0,
                min_end=0,
                end_comparator=-1,
            ),
            BaseScheduled(
                start_epoch=5.0,
                min_start=0,
                end_epoch=10.0,
                min_end=0,
                end_comparator=-1,
            ),
        ]
    )
    assert manager.min_epochs == 1.0
    assert manager.max_epochs == 10.0


def test_phase_dense_simple():
    manager = BaseManager(modifiers=[])
    assert manager.phase_at_end_of(epoch=0) == "dense"


def test_phase_dense_before_pruning():
    manager = BaseManager(modifiers=[_Pruning(start_epoch=10.0, end_epoch=20.0)])
    for epoch in range(10):
        assert manager.phase_at_end_of(epoch) == "dense"
    assert manager.phase_at_end_of(10) is None
    assert manager.phase_at_end_of(15) is None
    assert manager.phase_at_end_of(19) is None
    assert manager.phase_at_end_of(20) == "pruned"
    assert manager.phase_at_end_of(21) == "pruned"


def test_phase_dense_before_quantization():
    manager = BaseManager(modifiers=[_Quant(start_epoch=10.0, end_epoch=10.0)])
    for epoch in range(10):
        assert manager.phase_at_end_of(epoch) == "dense"
    assert manager.phase_at_end_of(10) == "dense_quantized"


def test_phase_dense_quantized_multiple():
    manager = BaseManager(
        modifiers=[
            _Quant(start_epoch=10.0, end_epoch=10.0),
            _Quant(start_epoch=20.0, end_epoch=20.0),
        ]
    )
    for epoch in range(20):
        assert manager.phase_at_end_of(epoch) != "dense_quantized"
    assert manager.phase_at_end_of(20) == "dense_quantized"


def test_phase_pruned_multiple():
    manager = BaseManager(
        modifiers=[
            _Pruning(start_epoch=10.0, end_epoch=20.0),
            _Pruning(start_epoch=25.0, end_epoch=30.0),
        ]
    )
    for epoch in range(30):
        assert manager.phase_at_end_of(epoch) != "pruned"
    assert manager.phase_at_end_of(30) == "pruned"


def test_phase_pruned_quant_single():
    manager = BaseManager(
        modifiers=[
            _Pruning(start_epoch=10.0, end_epoch=20.0),
            _Quant(start_epoch=25.0, end_epoch=25.0),
        ]
    )
    assert manager.phase_at_end_of(9) == "dense"
    assert manager.phase_at_end_of(10) is None
    assert manager.phase_at_end_of(20) == "pruned"
    assert manager.phase_at_end_of(24) == "pruned"
    assert manager.phase_at_end_of(25) == "pruned_quantized"


def test_phase_pruned_quant_multiple():
    manager = BaseManager(
        modifiers=[
            _Pruning(start_epoch=10.0, end_epoch=20.0),
            _Pruning(start_epoch=15.0, end_epoch=25.0),
            _Quant(start_epoch=35.0, end_epoch=35.0),
            _Quant(start_epoch=45.0, end_epoch=45.0),
        ]
    )
    for epoch in range(10):
        assert manager.phase_at_end_of(epoch) == "dense"
    for epoch in range(10, 25):
        assert manager.phase_at_end_of(epoch) is None
    for epoch in range(25, 35):
        assert manager.phase_at_end_of(epoch) == "pruned"
    for epoch in range(35, 45):
        assert manager.phase_at_end_of(epoch) is None
    assert manager.phase_at_end_of(45) == "pruned_quantized"


def test_phase_pruned_quant_overlap():
    manager = BaseManager(
        modifiers=[
            _Pruning(start_epoch=10.0, end_epoch=20.0),
            _Pruning(start_epoch=15.0, end_epoch=25.0),
            _Quant(start_epoch=25.0, end_epoch=25.0),
            _Quant(start_epoch=35.0, end_epoch=35.0),
        ]
    )
    for epoch in range(10):
        assert manager.phase_at_end_of(epoch) == "dense"
    for epoch in range(10, 35):
        assert manager.phase_at_end_of(epoch) is None
    assert manager.phase_at_end_of(35) == "pruned_quantized"


def test_phase_quant_pruned():
    manager = BaseManager(
        modifiers=[
            _Quant(start_epoch=10.0, end_epoch=10.0),
            _Quant(start_epoch=20.0, end_epoch=20.0),
            _Pruning(start_epoch=30.0, end_epoch=40.0),
            _Pruning(start_epoch=35.0, end_epoch=45.0),
        ]
    )
    for epoch in range(10):
        assert manager.phase_at_end_of(epoch) == "dense"
    for epoch in range(10, 20):
        assert manager.phase_at_end_of(epoch) is None
    for epoch in range(20, 30):
        assert manager.phase_at_end_of(epoch) == "dense_quantized"
    for epoch in range(30, 45):
        assert manager.phase_at_end_of(epoch) is None
    assert manager.phase_at_end_of(45) == "quantized_pruned"
