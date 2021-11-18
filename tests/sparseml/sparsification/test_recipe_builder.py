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

import random

import pytest

from sparseml.optim import BaseModifier, ModifierProp
from sparseml.sparsification import ModifierYAMLBuilder, RecipeYAMLBuilder, to_yaml_str


class FakeModifier(BaseModifier):
    def __init__(self, prop_one, prop_two):
        self._prop_one = prop_one
        self._prop_two = prop_two

    @ModifierProp()
    def prop_one(self):
        return self._prop_one

    @ModifierProp()
    def prop_two(self):
        return self._prop_two

    @ModifierProp(serializable=False)
    def hidden_prop(self):
        return f"{self._prop_one} {self._prop_two}"


@pytest.mark.parametrize(
    "modifier_class",
    [FakeModifier],
)
def test_modifier_builder_setters_getters(modifier_class):
    builder = ModifierYAMLBuilder(modifier_class)

    found_prop = False
    for attr in dir(modifier_class):
        prop = getattr(modifier_class, attr)
        if not isinstance(prop, ModifierProp):
            continue
        found_prop = True

        if prop.serializable:
            # check set and get supported
            assert getattr(builder, attr) is None
            random_val = random.random()
            setattr(builder, attr, random_val)
            assert getattr(builder, attr) == random_val
        else:
            # check set and get not supported
            with pytest.raises(ValueError):
                getattr(builder, attr)
            with pytest.raises(ValueError):
                setattr(builder, attr, random.random())

    assert found_prop  # ensure that modifier class is well formed and tests ran


def _create_fake_modifier_builder(prop_one, prop_two):
    return ModifierYAMLBuilder(FakeModifier, prop_one=prop_one, prop_two=prop_two)


def _expected_fake_modifier_yaml_str(prop_one, prop_two):
    prop_one = to_yaml_str(prop_one)
    prop_two = to_yaml_str(prop_two)
    return f"- !FakeModifier\n  prop_one: {prop_one}\n  prop_two: {prop_two}"


def _reduce_white_space(s):
    return " ".join(s.strip().split())


@pytest.mark.parametrize(
    "builder,expected_yaml_str",
    [
        (
            _create_fake_modifier_builder("val_1", 2),
            _expected_fake_modifier_yaml_str("val_1", 2),
        ),
        (
            _create_fake_modifier_builder({"key_1": [1, 2, "c"]}, [-1, 2, 3]),
            _expected_fake_modifier_yaml_str({"key_1": [1, 2, "c"]}, [-1, 2, 3]),
        ),
    ],
)
def test_modifier_builder_build_yaml(builder, expected_yaml_str):
    yaml_str = builder.build_yaml_str()

    yaml_str = _reduce_white_space(yaml_str)
    expected_yaml_str = _reduce_white_space(expected_yaml_str)

    assert yaml_str == expected_yaml_str


def _get_test_recipe_builder():
    return RecipeYAMLBuilder(
        variables={"init_lr": 0.01, "num_epochs": 100},
        modifier_groups={
            "test_modifiers": [
                _create_fake_modifier_builder(1, 2),
                _create_fake_modifier_builder("a", "b"),
            ]
        },
    )


_EXPECTED_TEST_RECIPE_YAML = """
init_lr: 0.01
num_epochs: 100

test_modifiers:
  - !FakeModifier
    prop_one: 1
    prop_two: 2

  - !FakeModifier
    prop_one: a
    prop_two: b
""".strip()


def test_recipe_builder_build_yaml_str():
    recipe_builder = _get_test_recipe_builder()
    recipe_yaml = recipe_builder.build_yaml_str()

    recipe_yaml = _reduce_white_space(recipe_yaml)
    expected_yaml = _reduce_white_space(_EXPECTED_TEST_RECIPE_YAML)

    assert recipe_yaml == expected_yaml


def test_recipe_builder_build_variables():
    recipe_builder = _get_test_recipe_builder()

    assert recipe_builder.has_variable("num_epochs")
    assert not recipe_builder.has_variable("target_sparsity")

    assert recipe_builder.get_variable("init_lr") == 0.01
    assert recipe_builder.get_variable("target_sparsity") is None

    recipe_builder.set_variable("init_lr", 0.05)
    builder_ref = recipe_builder.set_variable("target_sparsity", 0.9)
    assert builder_ref is recipe_builder  # test that setter is chainable

    assert recipe_builder.get_variable("init_lr") == 0.05
    assert recipe_builder.get_variable("target_sparsity") == 0.9


def test_recipe_builder_modifier_groups():
    recipe_builder = _get_test_recipe_builder()

    assert recipe_builder.get_modifier_group("modifiers") is None
    assert isinstance(recipe_builder.get_modifier_group("test_modifiers"), list)
    assert len(recipe_builder.get_modifier_group("test_modifiers")) == 2

    with pytest.raises(KeyError):
        # duplicate name
        recipe_builder.add_modifier_group("test_modifiers")

    with pytest.raises(ValueError):
        # group names must contain 'modifiers'
        recipe_builder.add_modifier_group("invalid_name")

    builder_ref = recipe_builder.add_modifier_group("modifiers")
    assert builder_ref is recipe_builder  # test method is chainable
    assert isinstance(recipe_builder.get_modifier_group("modifiers"), list)
    assert len(recipe_builder.get_modifier_group("modifiers")) == 0


def test_recipe_builder_get_modifier_builders():
    recipe_builder = _get_test_recipe_builder()

    assert len(recipe_builder.get_modifier_builders()) == 2
    assert len(recipe_builder.get_modifier_builders(modifier_type=FakeModifier)) == 2
    assert len(recipe_builder.get_modifier_builders(modifier_type=BaseModifier)) == 2
    assert len(recipe_builder.get_modifier_builders(modifier_type="FakeModifier")) == 2
    assert (
        len(
            recipe_builder.get_modifier_builders(
                modifier_type=FakeModifier, modifier_groups=["test_modifiers"]
            )
        )
        == 2
    )

    # filters that should produce no matches
    assert len(recipe_builder.get_modifier_builders(modifier_type="BaseModifier")) == 0
    assert len(recipe_builder.get_modifier_builders(modifier_type=str)) == 0
    assert len(recipe_builder.get_modifier_builders(modifier_groups=["modifiers"])) == 0
