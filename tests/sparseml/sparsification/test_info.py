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
import tempfile

import pytest

from sparseml.base import Framework
from sparseml.sparsification import (
    ModifierInfo,
    ModifierPropInfo,
    ModifierType,
    SparsificationInfo,
    load_sparsification_info,
    save_sparsification_info,
    sparsification_info,
)


def test_modifier_type():
    assert len(ModifierType) == 6
    assert ModifierType.general
    assert ModifierType.training
    assert ModifierType.pruning
    assert ModifierType.quantization
    assert ModifierType.act_sparsity
    assert ModifierType.misc


@pytest.mark.parametrize(
    "const_args",
    [
        {
            "name": "test name",
            "description": "test description",
            "type_": "str",
        },
        {
            "name": "test name",
            "description": "test description",
            "type_": "str",
            "restrictions": ["restriction"],
        },
    ],
)
def test_modifier_prop_info_lifecycle(const_args):
    # test construction
    info = ModifierPropInfo(**const_args)
    assert info, "No object returned for info constructor"

    # test serialization
    info_str = info.json()
    assert info_str, "No json returned for info"

    # test deserialization
    info_reconst = ModifierPropInfo.parse_raw(info_str)
    assert info == info_reconst, "Reconstructed does not equal original"


@pytest.mark.parametrize(
    "const_args",
    [
        {
            "name": "test name",
            "description": "test description",
        },
        {
            "name": "test name",
            "description": "test description",
            "type_": ModifierType.general,
            "props": [
                ModifierPropInfo(name="name", description="description", type_="str")
            ],
            "warnings": ["warning"],
        },
    ],
)
def test_modifier_info_lifecycle(const_args):
    # test construction
    info = ModifierInfo(**const_args)
    assert info, "No object returned for info constructor"

    # test serialization
    info_str = info.json()
    assert info_str, "No json returned for info"

    # test deserialization
    info_reconst = ModifierInfo.parse_raw(info_str)
    assert info == info_reconst, "Reconstructed does not equal original"


@pytest.mark.parametrize(
    "const_args",
    [
        {},
        {
            "modifiers": [
                ModifierInfo(
                    name="name",
                    description="description",
                    props=[
                        ModifierPropInfo(
                            name="name", description="description", type_="str"
                        )
                    ],
                )
            ]
        },
    ],
)
def test_sparsification_info_lifecycle(const_args):
    # test construction
    info = SparsificationInfo(**const_args)
    assert info, "No object returned for info constructor"

    # test serialization
    info_str = info.json()
    assert info_str, "No json returned for info"

    # test deserialization
    info_reconst = SparsificationInfo.parse_raw(info_str)
    assert info == info_reconst, "Reconstructed does not equal original"


def test_sparsification_info():
    # test that unknown raises an exception,
    # other sparsifications will test in their packages
    with pytest.raises(ValueError):
        sparsification_info(Framework.unknown)


def test_save_load_sparsification_info():
    info = SparsificationInfo(
        modifiers=[
            ModifierInfo(
                name="name",
                description="description",
                props=[
                    ModifierPropInfo(
                        name="name", description="description", type_="str"
                    )
                ],
            )
        ]
    )
    save_sparsification_info(info)
    loaded_json = load_sparsification_info(info.json())
    assert info == loaded_json

    test_path = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name
    save_sparsification_info(info, test_path)
    loaded_path = load_sparsification_info(test_path)
    assert info == loaded_path
    os.remove(test_path)
