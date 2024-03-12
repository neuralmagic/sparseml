from sparseml.transformers.compression import BitmaskConfig, DenseConfig, CompressionConfig
from sparseml.transformers.compression import BitmaskCompressor, ModelCompressor
import pytest


@pytest.mark.parametrize(
    "name,type",
    [
        ["sparse_bitmask", BitmaskConfig],
        ["dense", DenseConfig],
    ]
)
def test_configs(name, type):
    config = CompressionConfig.load_from_registry(name)
    assert isinstance(config, type)
    assert config.format == name

@pytest.mark.parametrize(
    "name,type",
    [
        ["sparse_bitmask", BitmaskCompressor],
    ]
)
def test_compressors(name, type):
    compressor = ModelCompressor.load_from_registry(name, config=CompressionConfig(format="none"))
    assert isinstance(compressor, type)
    assert isinstance(compressor.config, CompressionConfig)
    assert compressor.config.format == "none"