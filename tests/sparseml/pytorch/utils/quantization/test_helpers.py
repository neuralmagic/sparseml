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

import pytest
import torch

from sparseml.pytorch.models import mobilenet, resnet50
from sparseml.pytorch.utils.quantization import (
    QATWrapper,
    add_quant_dequant,
    configure_module_default_qconfigs,
    configure_module_qat_wrappers,
    fuse_module_conv_bn_relus,
    get_qat_qconfig,
)


try:
    from torch import quantization as torch_quantization
except Exception:
    torch_quantization = None


class _QATMatMul(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.wrap_qat = True
        self.qat_wrapper_kwargs = {
            "num_inputs": 2,
            "input_qconfigs": ["asymmetric", "symmetric"],
        }

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        return torch.matmul(a, b)


class _ModuleWrapper(torch.nn.Module):
    def __init__(self, module):
        super().__init__()

        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def _module_has_quant_stubs(module):
    return any(
        isinstance(submodule, torch_quantization.QuantStub)
        for submodule in module.modules()
    )


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
def test_configure_module_qat_wrappers():
    module = _ModuleWrapper(_ModuleWrapper(_QATMatMul()))

    assert not _module_has_quant_stubs(module)

    configure_module_qat_wrappers(module)

    qat_matmul = module.module.module

    assert isinstance(qat_matmul, QATWrapper)
    assert _module_has_quant_stubs(module)
    assert len(qat_matmul.input_quant_stubs) == 2
    assert len(qat_matmul.output_quant_stubs) == 1
    assert _count_submodule_instances(module, torch_quantization.QuantStub) == 3

    assert module(torch.randn(3, 3), torch.randn(3, 3)) is not None


def _assert_module_quant_stub_configs_exist(module, should_exist):
    for submodule in module.modules():
        if isinstance(submodule, torch_quantization.QuantStub):
            config_exists = (
                hasattr(submodule, "qconfig") and submodule.qconfig is not None
            )
            assert config_exists == should_exist


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
def test_configure_module_default_qconfigs():
    module = QATWrapper.from_module(_QATMatMul())
    _assert_module_quant_stub_configs_exist(module, False)

    configure_module_default_qconfigs(module)
    _assert_module_quant_stub_configs_exist(module, True)


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
def test_configure_module_default_qconfigs_no_config():
    module = QATWrapper.from_module(_QATMatMul())
    _assert_module_quant_stub_configs_exist(module, False)

    module.configure_qconfig = None

    configure_module_default_qconfigs(module)
    _assert_module_quant_stub_configs_exist(module, False)


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
