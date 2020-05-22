import pytest

import os
import torch
from torch import Tensor
from neuralmagicML.pytorch.recal import ModuleASAnalyzer

from tests.pytorch.helpers import MLPNet


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False), reason="Skipping pytorch tests",
)
def test_as_analyzer():
    model = MLPNet()
    layer_descs = [MLPNet.layer_descs()[1], MLPNet.layer_descs()[3]]
    analyzers = ModuleASAnalyzer.analyze_layers(
        model,
        [desc.name for desc in layer_descs],
        dim=None,
        track_inputs_sparsity=True,
        track_outputs_sparsity=True,
        inputs_sample_size=100,
        outputs_sample_size=100,
        enabled=True,
    )

    for _ in range(10):
        model(torch.randn(1, MLPNet.layer_descs()[0].input_size[0]))

    for analyzer in analyzers:
        assert isinstance(analyzer.inputs_sparsity_mean, Tensor)
        assert analyzer.inputs_sparsity_mean == 0
        assert isinstance(analyzer.inputs_sparsity_std, Tensor)
        assert analyzer.inputs_sparsity_std == 0

        assert isinstance(analyzer.inputs_sample_mean, Tensor)
        assert isinstance(analyzer.inputs_sample_std, Tensor)
        assert analyzer.inputs_sample_std > 0

        assert isinstance(analyzer.outputs_sparsity_mean, Tensor)
        assert analyzer.outputs_sparsity_mean > 0
        assert isinstance(analyzer.outputs_sparsity_std, Tensor)
        assert analyzer.outputs_sparsity_std > 0

        assert isinstance(analyzer.outputs_sample_mean, Tensor)
        assert analyzer.outputs_sample_mean > 0
        assert isinstance(analyzer.outputs_sample_std, Tensor)
        assert analyzer.outputs_sample_std > 0
