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
from typing import Any, Dict, List, NamedTuple, Union

import pytest

from flaky import flaky
from sparseml.onnx.optim.sensitivity_pruning import (
    PruningLossSensitivityAnalysis,
    pruning_loss_sens_magnitude,
    pruning_loss_sens_one_shot,
    pruning_perf_sens_one_shot,
)
from sparseml.onnx.utils.data import DataLoader
from tests.sparseml.onnx.helpers import GENERATE_TEST_FILES, OnnxRepoModelFixture


from tests.sparseml.onnx.helpers import onnx_repo_models  # noqa isort: skip


try:
    import deepsparse
except ModuleNotFoundError:
    deepsparse = None


RELATIVE_PATH = os.path.dirname(os.path.realpath(__file__))

OnnxModelAnalysisFixture = NamedTuple(
    "OnnxModelsAnalysisFixture",
    [
        ("model_path", str),
        ("input_paths", str),
        ("output_paths", str),
        ("loss_approx_path", str),
        ("loss_one_shot_path", str),
        ("perf_path", str),
        ("model_name", str),
        ("sparsity_levels", List[float]),
    ],
)


@pytest.fixture(scope="session")
def onnx_models_with_analysis(
    request, onnx_repo_models: OnnxRepoModelFixture  # noqa: F811
) -> OnnxModelAnalysisFixture:
    data_path = "test_sensitivity_ks_data"
    sparsity_levels = [0, 0.4, 0.8, 0.99]

    model_name = onnx_repo_models.model_name
    loss_approx_path = os.path.join(
        RELATIVE_PATH, data_path, "{}_loss_approx.json".format(model_name)
    )
    loss_one_shot_path = os.path.join(
        RELATIVE_PATH, data_path, "{}_loss_one_shot.json".format(model_name)
    )
    perf_path = os.path.join(
        RELATIVE_PATH, data_path, "{}_perf.json".format(model_name)
    )

    if GENERATE_TEST_FILES:
        _create_sensitivity_ks_data(
            onnx_repo_models.model_path,
            onnx_repo_models.input_paths,
            onnx_repo_models.output_paths,
            loss_approx_path,
            loss_one_shot_path,
            perf_path,
            sparsity_levels,
        )

    return OnnxModelAnalysisFixture(
        onnx_repo_models.model_path,
        onnx_repo_models.input_paths,
        onnx_repo_models.output_paths,
        loss_approx_path,
        loss_one_shot_path,
        perf_path,
        model_name,
        sparsity_levels,
    )


def _create_sensitivity_ks_data(
    model_path: str,
    input_paths: str,
    output_paths: str,
    loss_approx_path: str,
    loss_one_shot_path: str,
    perf_path: str,
    sparsity_levels: Union[List[float], None],
):
    dataloader = DataLoader(input_paths, output_paths, 1)
    analysis = pruning_loss_sens_magnitude(model_path)
    analysis.save_json(loss_approx_path)

    analysis = pruning_loss_sens_one_shot(
        model_path,
        dataloader,
        1,
        1,
        show_progress=False,
        sparsity_levels=sparsity_levels,
    )
    analysis.save_json(loss_one_shot_path)

    if deepsparse is not None:
        analysis = pruning_perf_sens_one_shot(
            model_path,
            dataloader,
            1,
            None,
            iterations_per_check=10,
            warmup_iterations_per_check=5,
            sparsity_levels=sparsity_levels,
            show_progress=False,
        )
        analysis.save_json(perf_path)


def _test_analysis_comparison(
    expected_layers: List[Dict[str, Any]],
    actual_layers: List[Dict[str, Any]],
    is_perf: bool,
):
    expected_layers_by_id = {layer["id"]: layer for layer in expected_layers}
    layer_pairs = [
        (expected_layers_by_id[actual_layer["id"]], actual_layer)
        for actual_layer in actual_layers
        if actual_layer["id"] in expected_layers_by_id
    ]  # List[(expected_layer, actual_layer)]

    if is_perf:
        # check that enough layers have a match that we can test engine perf groupings
        # may change between releases so may not always be able to line up all layers
        # consider updating golden files if these checks fail and perf results look as
        # expected
        assert len(layer_pairs) / len(actual_layers) >= 0.9
        assert abs(len(actual_layers) - len(expected_layers)) <= 5
    else:
        # check that all layers have a match
        assert len(layer_pairs) == len(actual_layers) == len(expected_layers)

    exact_equal_fields = ["id", "baseline_measurement_index", "has_baseline"]
    approximate_equal_fields = [
        ("baseline_average", 5e-3),
        ("sparse_average", 5e-3),
        ("sparse_integral", 5e-3),
        ("sparse_comparison", 5e-3),
    ]

    for expected_layer, actual_layer in layer_pairs:
        if actual_layer["id"] is None:
            continue  # cannot match None id
        for key in exact_equal_fields:
            if key not in expected_layer and key not in actual_layer:
                continue
            assert expected_layer[key] == actual_layer[key]

        for key, threshold in approximate_equal_fields:
            if (
                key not in expected_layer
                and key not in actual_layer
                or (not expected_layer[key] and not actual_layer[key])
            ):
                continue
            assert abs(expected_layer[key] - actual_layer[key]) < threshold

        for expected_sparsity, actual_sparsity in zip(
            expected_layer["averages"], actual_layer["averages"]
        ):
            assert abs(float(expected_sparsity) - float(actual_sparsity)) < 0.01
            assert (
                abs(
                    expected_layer["averages"][expected_sparsity]
                    - actual_layer["averages"][actual_sparsity]
                )
                < 5e-3
            )


@flaky(max_runs=2, min_passes=1)
def test_approx_ks_loss_sensitivity(
    onnx_models_with_analysis: OnnxModelAnalysisFixture,
):
    analysis = pruning_loss_sens_magnitude(onnx_models_with_analysis.model_path)

    expected_analysis = PruningLossSensitivityAnalysis.load_json(
        onnx_models_with_analysis.loss_approx_path
    )
    expected_layers = sorted(
        expected_analysis.dict()["results"], key=lambda x: x["index"]
    )
    actual_layers = sorted(analysis.dict()["results"], key=lambda x: x["index"])
    _test_analysis_comparison(expected_layers, actual_layers, False)


@flaky(max_runs=2, min_passes=1)
def test_one_shot_ks_loss_sensitivity(
    onnx_models_with_analysis: OnnxModelAnalysisFixture,
):
    dataloader = DataLoader(
        onnx_models_with_analysis.input_paths, onnx_models_with_analysis.output_paths, 1
    )

    analysis = pruning_loss_sens_one_shot(
        onnx_models_with_analysis.model_path,
        dataloader,
        1,
        1,
        show_progress=False,
        sparsity_levels=onnx_models_with_analysis.sparsity_levels,
    )
    expected_analysis = PruningLossSensitivityAnalysis.load_json(
        onnx_models_with_analysis.loss_one_shot_path
    )

    expected_layers = sorted(
        expected_analysis.dict()["results"], key=lambda x: x["index"]
    )

    actual_layers = sorted(
        analysis.dict()["results"],
        key=lambda x: x["index"],
    )

    _test_analysis_comparison(expected_layers, actual_layers, False)


@pytest.mark.skipif(
    deepsparse is None, reason="deepsparse is not installed on the system"
)
@flaky(max_runs=2, min_passes=1)
def test_one_shot_ks_perf_sensitivity(
    onnx_models_with_analysis: OnnxModelAnalysisFixture,
):
    expected_analysis = PruningLossSensitivityAnalysis.load_json(
        onnx_models_with_analysis.perf_path
    )

    dataloader = DataLoader(
        onnx_models_with_analysis.input_paths, onnx_models_with_analysis.output_paths, 1
    )

    analysis = pruning_perf_sens_one_shot(
        onnx_models_with_analysis.model_path,
        dataloader,
        1,
        None,
        iterations_per_check=10,
        warmup_iterations_per_check=3,
        sparsity_levels=onnx_models_with_analysis.sparsity_levels,
        show_progress=False,
    )

    expected_layers = sorted(
        expected_analysis.dict()["results"], key=lambda x: x["index"]
    )

    actual_layers = sorted(
        analysis.dict()["results"],
        key=lambda x: x["index"],
    )

    _test_analysis_comparison(expected_layers, actual_layers, True)
