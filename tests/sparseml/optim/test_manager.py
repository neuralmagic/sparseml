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
    assert manager.phase(epoch=0.0) == "dense"


def test_phase_dense_before_pruning():
    manager = BaseManager(modifiers=[_Pruning(start_epoch=10.0, end_epoch=20.0)])
    for epoch in [0.0, 5.0, 9.999]:
        assert manager.phase(epoch) == "dense"
    assert manager.phase(10.0) is None
    assert manager.phase(15.0) is None
    assert manager.phase(20.0) is None
    assert manager.phase(20.0001) == "pruned"


def test_phase_dense_before_quantization():
    manager = BaseManager(modifiers=[_Quant(start_epoch=10.0, end_epoch=10.0)])
    for epoch in [0.0, 5.0, 9.999]:
        assert manager.phase(epoch) == "dense"
    assert manager.phase(10.0) is None
    assert manager.phase(10.0001) == "dense_quantized"


def test_phase_dense_quantized_multiple():
    manager = BaseManager(
        modifiers=[
            _Quant(start_epoch=10.0, end_epoch=10.0),
            _Quant(start_epoch=20.0, end_epoch=20.0),
        ]
    )
    for epoch in [0.0, 5.0, 10.0, 15.0, 20.0]:
        assert manager.phase(epoch) != "dense_quantized"
    assert manager.phase(20.0001) == "dense_quantized"


def test_phase_pruned_multiple():
    manager = BaseManager(
        modifiers=[
            _Pruning(start_epoch=10.0, end_epoch=20.0),
            _Pruning(start_epoch=25.0, end_epoch=30.0),
        ]
    )
    for epoch in [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]:
        assert manager.phase(epoch) != "pruned"
    assert manager.phase(30.0001) == "pruned"


def test_phase_pruned_quant_single():
    manager = BaseManager(
        modifiers=[
            _Pruning(start_epoch=10.0, end_epoch=20.0),
            _Quant(start_epoch=25.0, end_epoch=25.0),
        ]
    )
    assert manager.phase(9.999) == "dense"
    assert manager.phase(10.0) is None
    assert manager.phase(20.0) is None
    assert manager.phase(20.0001) == "pruned"
    assert manager.phase(24.999) == "pruned"
    assert manager.phase(25.0) is None
    assert manager.phase(25.0001) == "pruned_quantized"


def test_phase_pruned_quant_multiple():
    manager = BaseManager(
        modifiers=[
            _Pruning(start_epoch=10.0, end_epoch=20.0),
            _Pruning(start_epoch=15.0, end_epoch=25.0),
            _Quant(start_epoch=35.0, end_epoch=35.0),
            _Quant(start_epoch=45.0, end_epoch=45.0),
        ]
    )
    assert manager.phase(9.999) == "dense"
    assert manager.phase(10.0) is None
    assert manager.phase(20.0) is None
    assert manager.phase(25.0) is None
    assert manager.phase(25.0001) == "pruned"
    assert manager.phase(30.0) == "pruned"
    assert manager.phase(35.0) is None
    assert manager.phase(40.0) is None
    assert manager.phase(45.0) is None
    assert manager.phase(45.0001) == "pruned_quantized"


def test_phase_pruned_quant_overlap():
    manager = BaseManager(
        modifiers=[
            _Pruning(start_epoch=10.0, end_epoch=20.0),
            _Pruning(start_epoch=15.0, end_epoch=25.0),
            _Quant(start_epoch=25.0, end_epoch=25.0),
            _Quant(start_epoch=35.0, end_epoch=35.0),
        ]
    )
    assert manager.phase(9.999) == "dense"
    assert manager.phase(10.0) is None
    assert manager.phase(20.0) is None
    assert manager.phase(25.0) is None
    assert manager.phase(25.0001) is None
    assert manager.phase(35.0) is None
    assert manager.phase(35.0001) == "pruned_quantized"


def test_phase_quant_pruned():
    manager = BaseManager(
        modifiers=[
            _Quant(start_epoch=10.0, end_epoch=10.0),
            _Quant(start_epoch=20.0, end_epoch=20.0),
            _Pruning(start_epoch=30.0, end_epoch=40.0),
            _Pruning(start_epoch=35.0, end_epoch=45.0),
        ]
    )
    assert manager.phase(9.999) == "dense"
    assert manager.phase(10.0) is None
    assert manager.phase(15.0) is None
    assert manager.phase(20.0) is None
    assert manager.phase(20.0001) == "dense_quantized"
    assert manager.phase(25.0) == "dense_quantized"
    assert manager.phase(29.999) == "dense_quantized"
    assert manager.phase(30.0) is None
    assert manager.phase(35.0) is None
    assert manager.phase(45.0) is None
    assert manager.phase(45.0001) == "quantized_pruned"
