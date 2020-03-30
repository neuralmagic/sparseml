import pytest

from typing import List
import os
from requests import HTTPError

from neuralmagicML.utils import available_models, RepoModel, models_download_file


@pytest.mark.parametrize(
    "mod,overwrite",
    [(mod, index == 0) for index, mod in enumerate(available_models()[:2])],
)
def test_available_models_downloads(mod: RepoModel, overwrite: bool):
    assert mod
    assert isinstance(mod, RepoModel)

    path = mod.download_onnx_file(overwrite)
    assert path
    assert os.path.exists(path)

    path = mod.download_framework_file(overwrite)
    assert path
    assert os.path.exists(path)


@pytest.mark.parametrize(
    "attr_key,filter_key,filter_vals",
    [
        ("domain", "domains", ["cv"]),
        ("sub_domain", "sub_domains", ["classification"]),
        ("architecture", "architectures", ["resnet-v1", "mobilenet-v1"]),
        ("sub_architecture", "sub_architectures", ["1.0", "50"]),
        ("framework", "frameworks", ["pytorch"]),
        ("desc", "descs", ["recal", "recal-perf"]),
    ],
)
def test_available_models_filters(
    attr_key: str, filter_key: str, filter_vals: List[str]
):
    available = available_models(**{filter_key: filter_vals})
    assert available

    for mod in available:
        assert isinstance(mod, RepoModel)
        val = mod.__getattribute__(attr_key)
        assert val in filter_vals


def test_download_failure():
    path = "does/not/exist.onnx"

    with pytest.raises(HTTPError):
        models_download_file(path, overwrite=True)
