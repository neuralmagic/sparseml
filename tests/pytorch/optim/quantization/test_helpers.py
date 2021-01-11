import os

import pytest
import torch
from sparseml.pytorch.models import mobilenet, resnet50
from sparseml.pytorch.optim.quantization import (
    add_quant_dequant,
    fuse_module_conv_bn_relus,
    get_qat_qconfig,
)


try:
    from torch import quantization as torch_quantization
except:
    torch_quantization = None


def _count_submodule_instances(module, clazz):
    return sum(1 for submodule in module.modules() if isinstance(submodule, clazz))


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_QUANT_TESTS", False),
    reason="Skipping pytorch torch quantization tests",
)
@pytest.mark.skipif(
    torch_quantization is None,
    reason="torch quantization not available",
)
@pytest.mark.parametrize(
    "model_lambda,num_quantizable_ops",
    [(mobilenet, 28), (resnet50, 54)],
)
def test_add_quant_dequant(model_lambda, num_quantizable_ops):
    module = model_lambda()
    module.qconfig = torch_quantization.default_qat_qconfig
    torch_quantization.propagate_qconfig_(module)
    add_quant_dequant(module)
    num_wrappers_added = _count_submodule_instances(
        module, torch_quantization.QuantWrapper
    )
    assert num_wrappers_added == num_quantizable_ops


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_QUANT_TESTS", False),
    reason="Skipping pytorch torch quantization tests",
)
@pytest.mark.skipif(
    torch_quantization is None,
    reason="torch quantization not available",
)
def test_get_qat_qconfig():
    assert isinstance(get_qat_qconfig(), torch_quantization.QConfig)


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_QUANT_TESTS", False),
    reason="Skipping pytorch torch quantization tests",
)
@pytest.mark.skipif(
    torch_quantization is None,
    reason="torch quantization not available",
)
@pytest.mark.parametrize(
    "model_lambda,conv_bn_relus,conv_bns",
    [(mobilenet, 27, 0), (resnet50, 45, 8)],
)
def test_fuse_module_conv_bn_relus(model_lambda, conv_bn_relus, conv_bns):
    module = model_lambda()
    conv_bn_relu_class = torch.nn.intrinsic.modules.fused.ConvBnReLU2d
    conv_bn_class = torch.nn.intrinsic.modules.fused.ConvBn2d

    # check model is not already fused
    assert _count_submodule_instances(module, conv_bn_relu_class) == 0
    assert _count_submodule_instances(module, conv_bn_class) == 0

    # fuse module and check expected number of fusions occured
    fuse_module_conv_bn_relus(module, inplace=True)
    assert _count_submodule_instances(module, conv_bn_relu_class) == conv_bn_relus
    assert _count_submodule_instances(module, conv_bn_class) == conv_bns
