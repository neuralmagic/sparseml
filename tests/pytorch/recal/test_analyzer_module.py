import pytest

from typing import Tuple
import torch
from torch.nn import Module
from torchvision.models import resnet50

from neuralmagicML.pytorch.recal import ModuleAnalyzer

from tests.pytorch.helpers import MLPNet, ConvNet


@pytest.mark.parametrize(
    "model,input_shape,name,params,prunable_params,execution_order,flops,total_flops",
    [
        (
            MLPNet(),
            MLPNet.layer_descs()[0].input_size,
            None,
            2800,
            2688,
            0,
            2912,
            5600,
        ),
        (
            MLPNet(),
            MLPNet.layer_descs()[0].input_size,
            MLPNet.layer_descs()[2].name,
            544,
            512,
            4,
            544,
            1056,
        ),
        (
            MLPNet(),
            MLPNet.layer_descs()[0].input_size,
            MLPNet.layer_descs()[3].name,
            0,
            0,
            5,
            32,
            32,
        ),
        (
            ConvNet(),
            ConvNet.layer_descs()[0].input_size,
            None,
            5418,
            5360,
            0,
            321780,
            632564,
        ),
        (
            ConvNet(),
            ConvNet.layer_descs()[0].input_size,
            ConvNet.layer_descs()[2].name,
            4640,
            4608,
            4,
            227360,
            453152,
        ),
        (
            resnet50(),
            (3, 224, 224),
            None,
            25557032,
            25502912,
            0,
            4140866536,
            8230050792,
        ),
    ],
)
def test_analyzer(
    model: Module,
    input_shape: Tuple[int],
    name: str,
    params: int,
    prunable_params: int,
    execution_order: int,
    flops: int,
    total_flops: int,
):
    analyzer = ModuleAnalyzer(model, enabled=True)
    tens = torch.randn(1, *input_shape)
    out = model(tens)
    analyzer.enabled = False
    out = model(tens)

    desc = analyzer.layer_desc(name)
    assert desc.params == params
    assert desc.prunable_params == prunable_params
    assert desc.execution_order == execution_order
    assert desc.flops == flops
    assert desc.total_flops == total_flops
