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

from sparseml.sparsification import (
    LayerInfo,
    ModelResult,
    PruningSensitivityResult,
    PruningSensitivityResultTypes,
    Result,
)


def _test_layer_info_eq(layer_one, layer_two):
    assert layer_one.name == layer_two.name
    assert layer_one.op_type == layer_two.op_type
    assert layer_one.params == layer_two.params
    assert layer_one.bias_params == layer_two.bias_params
    assert layer_one.prunable == layer_two.prunable
    assert layer_one.flops == layer_two.flops
    assert layer_one.execution_order == layer_two.execution_order
    assert layer_one.attributes == layer_two.attributes


@pytest.mark.parametrize(
    "layer_info,expected_dict",
    [
        (
            LayerInfo(name="layers.1", op_type="TestLayer", attributes={"val": 1}),
            {
                "name": "layers.1",
                "op_type": "TestLayer",
                "prunable": False,
                "execution_order": -1,
                "attributes": {"val": 1},
                "flops": None,
                "bias_params": None,
                "params": None,
            },
        ),
        (
            LayerInfo.linear_layer("layers.fc", 64, 128, True),
            {
                "name": "layers.fc",
                "op_type": "linear",
                "params": 64 * 128,
                "bias_params": 128,
                "prunable": True,
                "execution_order": -1,
                "attributes": {"in_channels": 64, "out_channels": 128},
                "flops": None,
            },
        ),
        (
            LayerInfo.conv_layer("layers.conv", 128, 256, [3, 3], False),
            {
                "name": "layers.conv",
                "op_type": "conv",
                "params": 128 * 256 * 3 * 3,
                "prunable": True,
                "execution_order": -1,
                "attributes": {
                    "in_channels": 128,
                    "out_channels": 256,
                    "kernel_shape": [3, 3],
                    "groups": 1,
                    "stride": 1,
                    "padding": [0, 0, 0, 0],
                },
                "bias_params": None,
                "flops": None,
            },
        ),
        (
            LayerInfo.conv_layer(
                "layers.depth.conv", 128, 256, [3, 3], False, groups=128
            ),
            {
                "name": "layers.depth.conv",
                "op_type": "conv",
                "params": 128 * 256 * 3 * 3 // 128,
                "prunable": True,
                "execution_order": -1,
                "attributes": {
                    "in_channels": 128,
                    "out_channels": 256,
                    "kernel_shape": [3, 3],
                    "groups": 128,
                    "stride": 1,
                    "padding": [0, 0, 0, 0],
                },
                "bias_params": None,
                "flops": None,
            },
        ),
    ],
)
def test_layer_info_serialization(layer_info, expected_dict):
    layer_info_dict = dict(layer_info)
    expected_dict_loaded = LayerInfo.parse_obj(expected_dict)
    layer_info_dict_reloaded = LayerInfo.parse_obj(layer_info_dict)

    assert type(expected_dict_loaded) is LayerInfo
    assert type(layer_info_dict_reloaded) is LayerInfo
    assert layer_info_dict == expected_dict
    _test_layer_info_eq(layer_info, expected_dict_loaded)
    _test_layer_info_eq(layer_info, layer_info_dict_reloaded)


def _result_eq(result_one, result_two) -> bool:
    return (
        result_one.value == result_two.value
        and result_one.attributes == result_two.attributes
    )


def _test_model_result_eq(result_one, result_two):
    assert _result_eq(result_one, result_two)
    assert result_one.analysis_type == result_two.analysis_type

    assert len(result_one.layer_results) == len(result_two.layer_results)
    for name_one, layer_result_one in result_one.layer_results.items():
        assert any(
            name_one == name_two and _result_eq(layer_result_one, layer_result_two)
            for name_two, layer_result_two in result_two.layer_results.items()
        )


@pytest.mark.parametrize(
    "model_result,expected_dict",
    [
        (
            ModelResult(analysis_type="lr_sensitivity", value={0.1: 100, 0.2: 50}),
            {
                "analysis_type": "lr_sensitivity",
                "value": {0.1: 100, 0.2: 50},
                "layer_results": {},
                "attributes": None,
            },
        ),
        (
            ModelResult(
                analysis_type="pruning_sensitivity",
                layer_results={
                    "net.1": Result(value={0.0: 0.25, 0.6: 0.2, 0.8: 0.1}),
                    "net.2": Result(value={0.0: 0.2, 0.6: 0.2, 0.8: 0.2}),
                },
            ),
            {
                "analysis_type": "pruning_sensitivity",
                "value": None,
                "layer_results": {
                    "net.1": {
                        "value": {0.0: 0.25, 0.6: 0.2, 0.8: 0.1},
                        "attributes": None,
                    },
                    "net.2": {
                        "value": {0.0: 0.2, 0.6: 0.2, 0.8: 0.2},
                        "attributes": None,
                    },
                },
                "attributes": None,
            },
        ),
    ],
)
def test_model_result_serialization(model_result, expected_dict):
    model_result_dict = model_result.dict()
    expected_dict_loaded = ModelResult.parse_obj(expected_dict)
    model_result_dict_reloaded = ModelResult.parse_obj(model_result_dict)

    assert type(expected_dict_loaded) is ModelResult
    assert type(model_result_dict_reloaded) is ModelResult

    for _, layer_result in expected_dict_loaded.layer_results.items():
        assert type(layer_result) is Result

    assert model_result_dict == expected_dict
    _test_model_result_eq(model_result, expected_dict_loaded)
    _test_model_result_eq(model_result, model_result_dict_reloaded)


def _fake_pruning_loss_sensitivity_result():
    result = PruningSensitivityResult(PruningSensitivityResultTypes.LOSS)
    result.add_layer_sparsity_result("layer_1", 0.8, 0.005)
    result.add_layer_sparsity_result("layer_1", 0.9, 0.003)
    result.add_layer_sparsity_result("layer_2", 0.8, 0.002)
    return result


def _fake_pruning_perf_sensitivity_result():
    result = PruningSensitivityResult(PruningSensitivityResultTypes.PERF)
    result.add_model_sparsity_result(0.8, 1.1)
    result.add_model_sparsity_result(0.9, 0.9)
    result.add_layer_sparsity_result("layer_1", 0.8, 0.5)
    result.add_layer_sparsity_result("layer_1", 0.9, 0.3)
    result.add_layer_sparsity_result("layer_2", 0.8, 0.2)
    return result


@pytest.mark.parametrize(
    "result,expected_dict",
    [
        (
            _fake_pruning_loss_sensitivity_result(),
            {
                "analysis_type": PruningSensitivityResultTypes.LOSS.value,
                "value": None,
                "layer_results": {
                    "layer_1": {
                        "value": {"0.8": 0.005, "0.9": 0.003},
                        "attributes": None,
                    },
                    "layer_2": {
                        "value": {"0.8": 0.002},
                        "attributes": None,
                    },
                },
                "attributes": None,
            },
        ),
        (
            _fake_pruning_perf_sensitivity_result(),
            {
                "analysis_type": PruningSensitivityResultTypes.PERF.value,
                "value": {
                    "0.8": 1.1,
                    "0.9": 0.9,
                },
                "layer_results": {
                    "layer_1": {
                        "value": {"0.8": 0.5, "0.9": 0.3},
                        "attributes": None,
                    },
                    "layer_2": {
                        "value": {"0.8": 0.2},
                        "attributes": None,
                    },
                },
                "attributes": None,
            },
        ),
    ],
)
def test_pruning_sensitivity_results(result, expected_dict):
    result_dict = result.dict()
    assert result_dict == expected_dict
    assert result_dict == PruningSensitivityResult.parse_obj(expected_dict).dict()
