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


import json
import os

import onnx
import pytest

from flaky import flaky
from sparseml.onnx.sparsification import (
    ModelInfo,
    PruningLossSensitivityMagnitudeAnalyzer,
    PruningPerformanceSensitivityAnalyzer,
)
from sparseml.sparsification import (
    PruningSensitivityResult,
    PruningSensitivityResultTypes,
)
from tests.sparseml.onnx.helpers import GENERATE_TEST_FILES, OnnxRepoModelFixture


from tests.sparseml.onnx.helpers import onnx_repo_models  # noqa isort: skip


try:
    import deepsparse
except Exception:
    deepsparse = None


RELATIVE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(RELATIVE_PATH, "test_analyzer_data")


def _load_pruning_analysis_test_results(path, live_results):
    if GENERATE_TEST_FILES:
        with open(path, "w") as live_results_file:
            live_results_json = live_results.dict()
            json.dump(live_results_json, live_results_file, indent=4)
    return PruningSensitivityResult.parse_file(path)


def _run_pruning_analysis(analyzer_class, model, **run_kwargs):
    model_info = ModelInfo(model)
    assert analyzer_class.available(model_info, model=model, **run_kwargs)

    analyzer = analyzer_class(model_info)
    return analyzer.run(
        model=model,
        show_progress=False,
        **run_kwargs,
    )


def _test_pruning_sensitivity_results(results, expected_results):
    assert results.analysis_type is not None
    assert results.analysis_type == expected_results.analysis_type

    if results.attributes:
        # model attributes
        assert set(results.attributes.keys()) == set(expected_results.attributes.keys())
        for key, value in results.attributes.items():
            if key == "num_cores":
                continue  # different systems may have different number of cores
            assert value == expected_results.attributes[key]

    if results.value is not None:
        # model level results
        assert expected_results.value is not None
        assert isinstance(results.value, dict)
        assert set(results.value.keys()) == set(expected_results.value.keys())

    # model layer results
    assert set(results.layer_results.keys()) == set(
        expected_results.layer_results.keys()
    )
    for layer, layer_results in results.layer_results.items():
        layer_results = layer_results.value
        expected_layer_results = expected_results.layer_results[layer].value
        assert set(layer_results.keys()) == set(expected_layer_results.keys())
        if results.analysis_type == PruningSensitivityResultTypes.PERF.value:
            # skip perf comparisons since tests may be run across systems
            continue
        for sparsity, sensitivity in layer_results.items():
            assert abs(sensitivity - expected_layer_results[sparsity]) < 5e-4


@flaky(max_runs=2, min_passes=1)
def test_pruning_loss_sensitivity_magnitude_analyzer(
    onnx_repo_models: OnnxRepoModelFixture,  # noqa: F811
):
    test_file_path = os.path.join(
        DATA_PATH, f"{onnx_repo_models.model_name}_pruning_loss_magnitude.json"
    )
    model = onnx.load(onnx_repo_models.model_path)

    results = _run_pruning_analysis(
        PruningLossSensitivityMagnitudeAnalyzer,
        model,
        pruning_loss_analysis_sparsity_levels=(0, 0.4, 0.8, 0.99),
    )
    expected_results = _load_pruning_analysis_test_results(test_file_path, results)

    _test_pruning_sensitivity_results(results, expected_results)


@pytest.mark.skipif(deepsparse is None, reason="unable to import deepsparse")
@flaky(max_runs=2, min_passes=1)
def test_pruning_performance_sensitivity_analyzer(
    onnx_repo_models: OnnxRepoModelFixture,  # noqa: F811
):
    test_file_path = os.path.join(
        DATA_PATH, f"{onnx_repo_models.model_name}_pruning_perf.json"
    )
    model = onnx.load(onnx_repo_models.model_path)

    results = _run_pruning_analysis(
        PruningPerformanceSensitivityAnalyzer,
        model,
        pruning_perf_analysis_sparsity_levels=(0, 0.4, 0.8, 0.99),
    )
    expected_results = _load_pruning_analysis_test_results(test_file_path, results)

    _test_pruning_sensitivity_results(results, expected_results)
