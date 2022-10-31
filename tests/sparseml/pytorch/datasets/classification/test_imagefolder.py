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

import builtins

import pytest


@pytest.fixture
def hide_torchvision(monkeypatch):
    import_orig = builtins.__import__

    def mocked_torchvision_import(name, *args, **kwargs):
        if name == "torchvision":
            raise ImportError()
        return import_orig(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mocked_torchvision_import)


@pytest.mark.usefixtures("hide_torchvision")
def test_import_error():
    with pytest.raises(ImportError, match="not found"):
        from sparseml.pytorch.datasets.classification import (  # noqa F401
            ImageFolderDataset,
        )
