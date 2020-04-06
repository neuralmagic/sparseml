import pytest

import tempfile
import torch

from neuralmagicML.pytorch.utils import ModuleExporter
from tests.pytorch.helpers import MLPNet


def test_exporter_onnx():
    sample_batch = torch.randn(1, 8)
    exporter = ModuleExporter(MLPNet(), tempfile.gettempdir())
    exporter.export_onnx(sample_batch)


@pytest.mark.parametrize("batch_size", [1, 64])
def test_export_batches(batch_size):
    sample_batch = torch.randn(batch_size, 8)
    exporter = ModuleExporter(MLPNet(), tempfile.gettempdir())
    exporter.export_samples([sample_batch])
