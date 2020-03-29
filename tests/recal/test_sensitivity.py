import pytest
import tempfile
import os

from neuralmagicML.recal import (
    KSLossSensitivityAnalysis,
    KSLossSensitivityResult,
    KSLossSensitivityProgress,
)


@pytest.mark.parametrize(
    "name,param_name,type_,execution_order,measured,expected_integral",
    [
        (
            "layer.name",
            "weight",
            "linear",
            0,
            [(0.05, 0.0), (0.25, 0.0), (0.5, 0.0), (0.75, 0.0), (0.95, 0.0)],
            0.0,
        ),
        (
            "layer.name",
            "weight",
            "conv",
            0,
            [(0.05, 1.0), (0.25, 1.0), (0.5, 1.0), (0.75, 1.0), (0.95, 1.0)],
            1.0,
        ),
        (
            "layer.name",
            "weight",
            "conv",
            0,
            [(0.05, 1.0), (0.25, 1.5), (0.5, 2.0), (0.75, 3.75), (0.95, 6.0)],
            2.7312498092651367,
        ),
    ],
)
def test_ks_loss_sensitivity_analysis_result(
    name, param_name, type_, execution_order, measured, expected_integral
):
    sens = KSLossSensitivityResult(name, param_name, type_, measured)
    assert sens.name == name
    assert sens.param_name == param_name
    assert sens.type_ == type_
    assert abs(sens.integral - expected_integral) < 0.0001

    analysis = KSLossSensitivityAnalysis()
    analysis.results.append(sens)

    assert sens == analysis.get_result(name)
    analysis.results_summary(normalize=False)
    analysis.results_summary(normalize=True)


@pytest.mark.parametrize(
    "name,param_name,type_,execution_order,measured,expected_integral",
    [
        (
            "layer.name",
            "weight",
            "linear",
            0,
            [(0.05, 0.0), (0.25, 0.0), (0.5, 0.0), (0.75, 0.0), (0.95, 0.0)],
            0.0,
        ),
        (
            "layer.name",
            "weight",
            "conv",
            0,
            [(0.05, 1.0), (0.25, 1.0), (0.5, 1.0), (0.75, 1.0), (0.95, 1.0)],
            1.0,
        ),
        (
            "layer.name",
            "weight",
            "conv",
            0,
            [(0.05, 1.0), (0.25, 1.5), (0.5, 2.0), (0.75, 3.75), (0.95, 6.0)],
            2.7312498092651367,
        ),
    ],
)
def test_ks_loss_sensitivity_analysis_load(
    name, param_name, type_, execution_order, measured, expected_integral
):
    sens = KSLossSensitivityResult(name, param_name, type_, measured)
    analysis = KSLossSensitivityAnalysis()
    analysis.results.append(sens)

    path = os.path.join(tempfile.gettempdir(), "ks-sens-analysis.json")
    analysis.save_json(path)

    json_analysis = analysis.load_json(path)

    assert len(json_analysis.results) == 1
    sens = json_analysis.results[0]
    assert sens.name == name
    assert sens.param_name == param_name
    assert sens.type_ == type_
    assert abs(sens.integral - expected_integral) < 0.0001

    path = os.path.join(tempfile.gettempdir(), "ks-sens-analysis-normalized.png")
    analysis.plot(path, normalize=True)
    assert os.path.exists(path)

    path = os.path.join(tempfile.gettempdir(), "ks-sens-analysis.png")
    analysis.plot(path, normalize=False)
    assert os.path.exists(path)


@pytest.mark.parametrize(
    "layer_index,layer_name,layers,sparsity_index,sparsity_levels,measurement_step,samples_per_measurement",
    [
        (
            0,
            "layer.name",
            ["layer.name", "layer2.name"],
            0,
            [0.2, 0.5, 0.7, 0.9, 0.95],
            0,
            100,
        )
    ],
)
def test_ks_loss_sensitivity_progress(
    layer_index,
    layer_name,
    layers,
    sparsity_index,
    sparsity_levels,
    measurement_step,
    samples_per_measurement,
):
    progress = KSLossSensitivityProgress(
        layer_index,
        layer_name,
        layers,
        sparsity_index,
        sparsity_levels,
        measurement_step,
        samples_per_measurement,
    )
    assert progress.layer_index == layer_index
    assert progress.layer_name == layer_name
    assert progress.layers == layers
    assert progress.sparsity_index == sparsity_index
    assert progress.sparsity_levels == sparsity_levels
    assert progress.measurement_step == measurement_step
    assert progress.samples_per_measurement == samples_per_measurement

    hook = KSLossSensitivityProgress.standard_update_hook()
    hook(progress)
    hook(progress)
