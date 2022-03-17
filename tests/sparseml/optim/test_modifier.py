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

from typing import Callable, Dict

import pytest

from sparseml.optim import (
    BaseModifier,
    BaseObject,
    BaseProp,
    BaseScheduled,
    BaseUpdate,
    ModifierProp,
    ModifierYAML,
)


__all__ = ["BaseModifierTest", "BaseScheduledTest", "BaseUpdateTest"]


class BaseModifierTest(object):
    def initialize_helper(self, modifier: BaseModifier, **kwargs):
        modifier._initialized = True

    def test_constructor(
        self,
        modifier_lambda: Callable[[], BaseModifier],
        framework: str,
        *args,
        **kwargs,
    ):
        modifier = modifier_lambda()
        assert isinstance(modifier, BaseModifier)

    def test_yaml(
        self,
        modifier_lambda: Callable[[], BaseModifier],
        framework: str,
        *args,
        **kwargs,
    ):
        modifier = modifier_lambda()
        mod_string = str(modifier)
        assert modifier.__class__.__name__ in mod_string
        assert framework not in mod_string

        # test lists
        mod_yaml_list = BaseModifier.load_framework_list(mod_string, framework)
        assert len(mod_yaml_list) == 1
        assert mod_yaml_list[0] is not None
        assert isinstance(mod_yaml_list[0], BaseModifier)
        assert isinstance(mod_yaml_list[0], modifier.__class__)

        # test obj
        mod_yaml_obj = BaseModifier.load_framework_obj(mod_string, framework)
        assert mod_yaml_obj is not None
        assert isinstance(mod_yaml_obj, BaseModifier)
        assert isinstance(mod_yaml_obj, modifier.__class__)

    def test_yaml_key(
        self,
        modifier_lambda: Callable[[], BaseModifier],
        framework: str,
        *args,
        **kwargs,
    ):
        modifier = modifier_lambda()
        mod_yaml_key = BaseModifier.yaml_key(modifier.__class__, framework)

        assert mod_yaml_key.startswith("!")
        assert framework in mod_yaml_key
        assert modifier.__class__.__name__ in mod_yaml_key

    def test_repr(
        self,
        modifier_lambda: Callable[[], BaseModifier],
        framework: str,
        *args,
        **kwargs,
    ):
        modifier = modifier_lambda()
        mod_repr = repr(modifier)

        assert modifier.__class__.__name__ in mod_repr
        assert framework not in mod_repr

    def test_props(
        self,
        modifier_lambda: Callable[[], BaseModifier],
        framework: str,
        initialize_kwargs: Dict = None,
        *args,
        **kwargs,
    ):
        modifier = modifier_lambda()

        expected_serializable = []
        expected_all = ["initialized", "enabled"]

        found_all = {}
        props = {}
        serializable = []
        restricted = []

        for attr in dir(modifier.__class__):
            prop = getattr(modifier.__class__, attr)

            if not isinstance(prop, ModifierProp):
                continue

            props[attr] = prop
            val = getattr(modifier, attr)

            if prop._func_set:
                setattr(modifier, attr, val)

            found_all[attr] = val

            if prop.serializable:
                serializable.append(attr)

            if "_initialized" in prop.restrictions:
                restricted.append(attr)

        for exp in expected_serializable:
            assert exp in serializable

        for exp in expected_all:
            assert exp in found_all

        if initialize_kwargs:
            self.initialize_helper(modifier, **initialize_kwargs)
        else:
            self.initialize_helper(modifier)

        for attr, val in found_all.items():
            prop = props[attr]

            if attr in restricted and prop._func_set:
                with pytest.raises(AttributeError):
                    setattr(modifier, attr, val)


class BaseScheduledTest(object):
    def test_props_start(
        self,
        modifier_lambda: Callable[[], BaseModifier],
        framework: str,
        *args,
        **kwargs,
    ):
        modifier = modifier_lambda()  # type: BaseScheduled

        assert "start_epoch" in dir(modifier.__class__)
        assert "start_epoch" in dir(modifier)
        prop = getattr(modifier.__class__, "start_epoch")
        assert prop.serializable
        assert "_initialized" in prop.restrictions

        val = modifier.start_epoch
        modifier.start_epoch = val
        modifier.start_epoch = modifier._min_start

        with pytest.raises(ValueError):
            modifier.start_epoch = modifier._min_start - 1.0

    def test_props_end(
        self,
        modifier_lambda: Callable[[], BaseModifier],
        framework: str,
        *args,
        **kwargs,
    ):
        modifier = modifier_lambda()  # type: BaseScheduled

        assert "end_epoch" in dir(modifier.__class__)
        assert "end_epoch" in dir(modifier)
        prop = getattr(modifier.__class__, "end_epoch")
        assert prop.serializable
        assert "_initialized" in prop.restrictions

        modifier.end_epoch

        with pytest.raises(ValueError):
            modifier.end_epoch = modifier._min_end - 1.0

        if modifier._end_comparator is not None:
            modifier.end_epoch = max(modifier._min_end, modifier.start_epoch) + 1.0
        elif modifier._end_comparator == -1:
            modifier.end_epoch = modifier._min_end
            modifier.end_epoch = modifier.start_epoch

            if modifier.start_epoch - 0.1 >= modifier._min_end:
                modifier.end_epoch = modifier.start_epoch - 0.1
        elif modifier._end_comparator == 0:
            if modifier.start_epoch >= modifier._min_end:
                modifier.end_epoch = modifier.start_epoch
            else:
                modifier.end_epoch = modifier._min_end


class BaseUpdateTest(object):
    def test_props_frequency(
        self,
        modifier_lambda: Callable[[], BaseModifier],
        framework: str,
        *args,
        **kwargs,
    ):
        modifier = modifier_lambda()  # type: BaseUpdate

        assert "update_frequency" in dir(modifier.__class__)
        assert "update_frequency" in dir(modifier)
        prop = getattr(modifier.__class__, "update_frequency")
        assert prop.serializable
        assert "_initialized" in prop.restrictions

        val = modifier.update_frequency
        modifier.update_frequency = val
        modifier.update_frequency = modifier._min_frequency

        with pytest.raises(ValueError):
            modifier.update_frequency = modifier._min_frequency - 1.0


def test_base_prop():
    class BasePropImpl(BaseProp):
        def __get__(self, obj, obj_type=None):
            pass

        def __set__(self, obj, value):
            pass

        def getter(self, func_get: Callable):
            pass

        def setter(self, func_set: Callable):
            pass

    impl = BasePropImpl()
    impl.__get__(None)
    impl.__set__(None, None)

    with pytest.raises(AttributeError):
        impl.__delete__(None)


def test_modifier_prop_attributes():
    prop = ModifierProp()
    assert prop.serializable
    prop = ModifierProp(serializable=False)
    assert not prop.serializable

    prop = ModifierProp()
    assert "_initialized" in prop.restrictions
    prop = ModifierProp(restrict_initialized=False)
    assert "_initialized" not in prop.restrictions

    prop = ModifierProp()
    assert "_enabled" not in prop.restrictions
    prop = ModifierProp(restrict_enabled=True)
    assert "_enabled" in prop.restrictions

    prop = ModifierProp(restrict_initialized=False, restrict_enabled=False)
    assert len(prop.restrictions) == 0
    prop = ModifierProp(
        restrict_initialized=False, restrict_enabled=False, restrict_extras=["_extra"]
    )
    assert len(prop.restrictions) == 1
    assert "_extra" in prop.restrictions


def test_modifier_prop_modifiers():
    prop = ModifierProp()

    with pytest.raises(AttributeError):
        prop.__get__({})

    with pytest.raises(AttributeError):
        prop.__set__({}, None)

    with pytest.raises(AttributeError):
        prop.__delete__({})

    obj = {"val": 1}

    def _getter(self):
        return self["val"]

    def _setter(self, val):
        self["val"] = val

    prop = ModifierProp()
    prop = prop.getter(_getter)

    assert prop.__get__(obj) == 1

    with pytest.raises(AttributeError):
        prop.__set__({}, None)

    with pytest.raises(AttributeError):
        prop.__delete__({})

    prop = prop.setter(_setter)
    prop.__set__(obj, 2)
    assert prop.__get__(obj) == 2

    with pytest.raises(AttributeError):
        prop.__delete__({})


def test_modifier_prop_class():
    class ModifierPropImpl(object):
        def __init__(self):
            self._initialized = False
            self._enabled = False
            self._extra = False

            self._val = -1
            self._get_val = -2

            self._init_val = 1
            self._enabled_val = 2
            self._extra_val = 3

        @ModifierProp(restrict_initialized=False)
        def val(self):
            return self._val

        @val.setter
        def val(self, value):
            self._val = value

        @ModifierProp()
        def get_val(self):
            return self._get_val

        @ModifierProp()
        def init_val(self):
            return self._init_val

        @init_val.setter
        def init_val(self, value):
            self._init_val = value

        @ModifierProp(restrict_initialized=False, restrict_enabled=True)
        def enabled_val(self):
            return self._enabled_val

        @enabled_val.setter
        def enabled_val(self, value):
            self._enabled_val = value

        @ModifierProp(restrict_initialized=False, restrict_extras=["_extra"])
        def extra_val(self):
            return self._extra_val

        @extra_val.setter
        def extra_val(self, value):
            self._extra_val = value

    prop_test = ModifierPropImpl()

    # test getters equal the constructed value
    assert prop_test.val == -1
    assert prop_test.get_val == -2
    assert prop_test.init_val == 1
    assert prop_test.enabled_val == 2
    assert prop_test.extra_val == 3

    # test setters pass / fail with no truthy values
    with pytest.raises(AttributeError):
        prop_test.get_val = -20
    prop_test.val = -10
    prop_test.init_val = 10
    prop_test.enabled_val = 20
    prop_test.extra_val = 30
    assert prop_test.get_val == -2
    assert prop_test.val == -10
    assert prop_test.init_val == 10
    assert prop_test.enabled_val == 20
    assert prop_test.extra_val == 30

    prop_test._initialized = True
    with pytest.raises(AttributeError):
        prop_test.get_val = -200
    prop_test.val = -100
    with pytest.raises(AttributeError):
        prop_test.init_val = 100
    prop_test.enabled_val = 200
    prop_test.extra_val = 300
    assert prop_test.get_val == -2
    assert prop_test.val == -100
    assert prop_test.init_val == 10
    assert prop_test.enabled_val == 200
    assert prop_test.extra_val == 300

    prop_test._initialized = False
    prop_test._enabled = True
    with pytest.raises(AttributeError):
        prop_test.get_val = -2000
    prop_test.val = -1000
    prop_test.init_val = 1000
    with pytest.raises(AttributeError):
        prop_test.enabled_val = 2000
    prop_test.extra_val = 3000
    assert prop_test.get_val == -2
    assert prop_test.val == -1000
    assert prop_test.init_val == 1000
    assert prop_test.enabled_val == 200
    assert prop_test.extra_val == 3000

    prop_test._enabled = False
    prop_test._extra = True
    with pytest.raises(AttributeError):
        prop_test.get_val = -20000
    prop_test.val = -10000
    prop_test.init_val = 10000
    prop_test.enabled_val = 20000
    with pytest.raises(AttributeError):
        prop_test.extra_val = 30000
    assert prop_test.get_val == -2
    assert prop_test.val == -10000
    assert prop_test.init_val == 10000
    assert prop_test.enabled_val == 20000
    assert prop_test.extra_val == 3000


def test_base_object():
    BaseObject()

    with pytest.raises(ValueError):
        BaseObject(one=1)

    with pytest.raises(TypeError):
        BaseObject(1)


@ModifierYAML("test_framework")
class BaseModifierImpl(BaseModifier):
    def __init__(self):
        super().__init__()


@pytest.mark.parametrize("modifier_lambda", [lambda: BaseModifierImpl()])
@pytest.mark.parametrize("framework", ["test_framework"])
class TestBaseModifier(BaseModifierTest):
    pass


class BaseScheduledImpl(BaseScheduled):
    def __init__(
        self,
        start_epoch: float = 0.0,
        min_start: float = -1.0,
        end_epoch: float = 1.0,
        min_end: float = -1.0,
        end_comparator: int = 0,
        **kwargs,
    ):
        super().__init__(
            start_epoch, min_start, end_epoch, min_end, end_comparator, **kwargs
        )


@pytest.mark.parametrize("modifier_lambda", [lambda: BaseScheduledImpl()])
@pytest.mark.parametrize("framework", ["test_framework"])
class TestBaseScheduled(BaseScheduledTest):
    pass


class BaseUpdateImpl(BaseUpdate):
    def __init__(
        self, update_frequency: float = 0.0, min_frequency: float = -1.0, **kwargs
    ):
        super().__init__(update_frequency, min_frequency, **kwargs)


@pytest.mark.parametrize("modifier_lambda", [lambda: BaseUpdateImpl()])
@pytest.mark.parametrize("framework", ["test_framework"])
class TestBaseUpdate(BaseUpdateTest):
    pass


def test_modifier_yaml_const():
    ModifierYAML("framework")
    ModifierYAML(framework="framework")

    with pytest.raises(TypeError):
        ModifierYAML()

    with pytest.raises(ValueError):
        ModifierYAML("")


def test_modifier_yaml_class():
    framework = "framework"

    @ModifierYAML(framework)
    class ModifierYAMLImpl(object):
        def __init__(self, arg_one, arg_two):
            self._arg_one = arg_one
            self._arg_two = arg_two

    yaml_str = """
    !{}.{}
        arg_one: 1
        arg_two: 2
    """.format(
        framework, ModifierYAMLImpl.__name__
    )
    obj = BaseModifier.load_framework_obj(yaml_str, framework)
    assert obj._arg_one == 1
    assert obj._arg_two == 2
