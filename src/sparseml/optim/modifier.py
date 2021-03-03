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

"""
Code related to modifiers that is shared across frameworks.
Modifiers allow modifying the training process of a model; ex to perform model pruning.
"""

import re
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Union

import yaml

from sparseml.utils import validate_str_iterable


__all__ = [
    "BaseProp",
    "ModifierProp",
    "BaseObject",
    "BaseModifier",
    "BaseScheduled",
    "BaseUpdate",
    "ModifierYAML",
]


class BaseProp(ABC):
    """
    BaseProp class meant to be implemented by any property decorators
    """

    @abstractmethod
    def __get__(self, obj, obj_type=None):
        """
        :param obj: the object to get the attribute from
        :param obj_type: unused
        :return: The retrieved value from the obj
        """
        pass

    @abstractmethod
    def __set__(self, obj, value):
        """
        :param obj: the object to get the attribute from
        :param value: the value to set
        """
        pass

    def __delete__(self, obj):
        """
        Override to not allow deletes for modifier properties

        :param obj: the object
        """
        raise AttributeError("can't delete attribute")

    @abstractmethod
    def getter(self, func_get: Callable):
        """
        :param func_get: the getter function
        :return: the recreated instance with the new getter function
        """
        pass

    @abstractmethod
    def setter(self, func_set: Callable):
        """
        :param func_set: the setter function
        :return: the recreated instance with the new setter function
        """
        pass


class ModifierProp(BaseProp):
    """
    Property used to decorate a modifier.
    Use for creating getters and setters in a modifier.
    Handles making sure props cannot be changed after a certain point;
    ex after initialized.
    Also, marks the properties so they can be easily collected and serialized later.

    :param serializable: True if the property should be serialized (ex in yaml),
        False otherwise. Default True
    :param restrict_initialized: True to keep the property from being set after
        initialized, False otherwise. Default True
    :param restrict_enabled: True to keep the property from being set after enabled,
        False otherwise. Default False
    :param restrict_extras: extra attributes to check, if any are truthy then keep
        from being set. Default None
    :param no_serialize_val: If prop is equal to this value, will not serialize the prop
    :param func_get: The function getter
    :param func_set: The function setter
    :param doc: The docs function
    """

    def __init__(
        self,
        serializable: bool = True,
        restrict_initialized: bool = True,
        restrict_enabled: bool = False,
        restrict_extras: List[str] = None,
        no_serialize_val: Any = None,
        func_get: Callable = None,
        func_set: Callable = None,
        doc: Callable = None,
    ):
        self._func_get = func_get
        self._func_set = func_set
        self._serializable = serializable
        self._restrictions = []
        self._no_serialize_val = no_serialize_val

        if restrict_initialized:
            self._restrictions.append("_initialized")

        if restrict_enabled:
            self._restrictions.append("_enabled")

        if restrict_extras is not None:
            self._restrictions.extend(restrict_extras)

        if doc is None and self._func_get is not None:
            doc = self._func_get.__doc__

        self.__doc__ = doc

    def __call__(self, getter: Callable) -> BaseProp:
        """
        :param getter: the annotated getter to use to get the attribute
        :return: the current property instance
        """
        self._func_get = getter

        return self

    def __get__(self, obj, obj_type=None):
        """
        Get the attribute from the current given object for the current modifier

        :param obj: the object to get the attribute from
        :param obj_type: unused
        :return: The retrieved value from the obj
        """
        if obj is None:
            return self

        if self._func_get is None:
            raise AttributeError("unreadable attribute")

        return self._func_get(obj)

    def __set__(self, obj, value):
        """
        Set the attribute in the current given object for the current modifier.
        If the attribute can't be set because of the current modifiers state,
        (ex: initialized) then will raise a AttributeError

        :param obj: the object to get the attribute from
        :param value: the value to set
        """
        if self._func_set is None:
            raise AttributeError("can't set attribute")

        if self.restrictions:
            for rest in self.restrictions:
                if hasattr(obj, rest) and getattr(obj, rest):
                    raise AttributeError(
                        "Cannot change {} after initializing {}".format(
                            self._func_get.__name__, obj.__class__.__name__
                        )
                    )

        self._func_set(obj, value)

    @property
    def serializable(self) -> bool:
        """
        :return: True if the property should be serialized (ex in yaml), False otherwise
        """
        return self._serializable

    @property
    def restrictions(self) -> List[str]:
        """
        :return: The attributes to check for restricting when the attribute can be set
        """
        return self._restrictions

    @property
    def no_serialize_val(self) -> Any:
        """
        :return: a value that if the prop is equal to, will not serialize the prop
        """
        return self._no_serialize_val

    def getter(self, func_get: Callable) -> BaseProp:
        """
        Create a ModifierProp based off the current instance with the getter function

        :param func_get: the getter function
        :return: the recreated instance with the new getter function
        """
        return type(self)(
            **self._creator_kwargs(),
            func_get=func_get,
            func_set=self._func_set,
            doc=self.__doc__,
        )

    def setter(self, func_set: Callable) -> BaseProp:
        """
        Create a ModifierProp based off the current instance with the setter function

        :param func_set: the setter function
        :return: the recreated instance with the new setter function
        """
        return type(self)(
            **self._creator_kwargs(),
            func_get=self._func_get,
            func_set=func_set,
            doc=self.__doc__,
        )

    def _creator_kwargs(self) -> Dict:
        return {
            "serializable": self._serializable,
            "restrict_initialized": False,
            "restrict_enabled": False,
            "restrict_extras": self._restrictions,
        }


class BaseObject(object):
    """
    BaseObject to accept kwargs so multiple inheritance will work with
    the modifier classes.
    kwargs param must be empty by the time this class is called.

    :param kwargs: standard key word args, used to support multi inheritance
    """

    def __init__(self, **kwargs):
        super().__init__()

        if len(kwargs) != 0:
            raise ValueError(
                (
                    "kwargs must be empty at _BaseObject, "
                    "extras passed in of {} for {}"
                ).format(kwargs, self.__class__.__name__)
            )


class BaseModifier(BaseObject):
    """
    Parent class meant to be used for all modifiers.
    Handles base implementations for properties and methods.

    :param log_types: the loggers that can be used by the modifier instance
    :param kwargs: standard key word args, used to support multi inheritance
    """

    @staticmethod
    def _convert_to_framework_modifiers(yaml_str: str, framework: str) -> str:
        pattern = re.compile(r"!(?P<mod_class>(?!.*\.)[a-zA-Z_][a-zA-Z^._0-9]+)")
        yaml_str = pattern.sub(r"!{}.\g<mod_class>".format(framework), yaml_str)

        return yaml_str

    @staticmethod
    def load_framework_list(yaml_str: str, framework: str):
        """
        :param yaml_str: a string representation of the yaml syntax to load modifiers
        :param framework: the framework to load the modifiers for
        :return: the loaded modifiers list
        """
        yaml_str = BaseModifier._convert_to_framework_modifiers(yaml_str, framework)
        container = yaml.safe_load(yaml_str)

        if isinstance(container, BaseModifier):
            modifiers = [container]
        elif isinstance(container, List):
            modifiers = container
        else:  # Dict
            modifiers = []
            for name, item in container.items():
                if "modifiers" in name and isinstance(item, List):
                    modifiers.extend(item)
                elif isinstance(item, BaseModifier):
                    modifiers.append(item)
                elif isinstance(item, List) and any(
                    isinstance(element, BaseModifier) for element in item
                ):
                    modifier_type = type(
                        [mod for mod in item if isinstance(mod, BaseModifier)][0]
                    )
                    raise ValueError(
                        "Invalid modifier location. Grouped modifiers in recipes must "
                        "be listed in lists with 'modifiers' in its name. A modifier of "
                        f"type {modifier_type} was found in recipe list {name}"
                    )

        return modifiers

    @staticmethod
    def load_framework_obj(yaml_str: str, framework: str):
        """
        :param yaml_str:  a string representation of the yaml syntax to load a modifier
        :param framework: the framework to load the modifier for
        :return: the loaded modifier object
        """
        yaml_str = BaseModifier._convert_to_framework_modifiers(yaml_str, framework)
        modifier = yaml.safe_load(yaml_str)

        return modifier

    @staticmethod
    def yaml_key(clazz, framework: Union[str, None] = None):
        """
        create a key for loading in yaml from the class and the framework

        :param clazz: the class representation to create the key for
        :param framework: the string representing the ML framework the modifier class
            is for. Default is None.
        :return: the formatted key; ex: !{framework}.{clazz.__name__}
        """
        if framework is None:
            return "!{}".format(clazz.__name__)

        return "!{}.{}".format(framework, clazz.__name__)

    def __init__(self, log_types: Union[str, List[str]], **kwargs):
        super().__init__(**kwargs)
        self._log_types = (
            validate_str_iterable(
                log_types, "log_types for {}".format(self.__class__.__name__)
            )
            if log_types
            else None
        )
        self._initialized = False
        self._enabled = True

    def __str__(self):
        formatted = [
            "    {}".format("{}: {}".format(key, val))
            for key, val in self.props(only_serializable=True, format_str=True).items()
        ]

        return "{}\n{}".format(
            BaseModifier.yaml_key(self.__class__), "\n".join(formatted)
        )

    def __repr__(self):
        return "{}({})".format(
            self.__class__.__name__,
            self.props(only_serializable=False, format_repr=True),
        )

    @ModifierProp(serializable=True)
    def log_types(self) -> Union[None, str, List[str]]:
        """
        :return: the loggers that can be used by the modifier instance
        """
        return self._log_types

    @ModifierProp(serializable=False, restrict_initialized=False)
    def initialized(self) -> bool:
        """
        :return: True if the modifier has gone through the initialized life cycle,
            False otherwise
        """
        return self._initialized

    @ModifierProp(serializable=False, restrict_initialized=False)
    def enabled(self) -> bool:
        """
        :return: True if the modifier is currently enabled and making updates,
            False otherwise
        """
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        """
        :param value: True to allow the modifier to make updates, False otherwise
        """
        self._enabled = value

    def props(
        self,
        only_serializable: bool,
        format_str: bool = False,
        format_repr: bool = False,
    ) -> Dict[str, Any]:
        """
        Gather all the ModifierProps for the current instance into a dictionary
        collection.

        :param only_serializable: True if only props marked as serializable should
            be collected, False otherwise
        :param format_str: True to format the values properly for a str.
            Ex: None values are formatted to null and otherwise str is called
        :param format_repr: True to format the values properly for a repr.
        :return: the collected properties with names mapping to values
        """
        if format_str and format_repr:
            raise ValueError(
                "only format_str or format_repr can be True, both are currently True"
            )

        props = {}

        for attr in dir(self):
            if attr.startswith("_"):
                continue

            func = getattr(self.__class__, attr)

            if not isinstance(func, ModifierProp) or (
                only_serializable and not func.serializable
            ):
                continue

            val = getattr(self, attr)
            no_serialize_val = func.no_serialize_val

            if val == no_serialize_val:
                continue

            if format_str:
                props[attr] = str(val) if val is not None else "null"
            elif format_repr:
                props[attr] = repr(val)
            else:
                props[attr] = val

        return props


class BaseScheduled(BaseObject):
    """
    Abstract class meant to be used for all scheduled modifiers.
    :py:func `~Modifier` is also meant to be inherited alongside this class.
    Handles base implementations for scheduled properties and methods to allow
    a schedule to be enforced.

    :param start_epoch: the epoch to start the scheduled modifier at
    :param min_start: the minimum value start_epoch can be,
        otherwise will raise a ValueError
    :param end_epoch: the epoch to end the scheduled modifier at
    :param min_end: the minimum value end_epoch can be,
     otherwise will raise a ValueError
    :param end_comparator: integer value representing how the end_epoch should be
        compared to start_epoch.
        if == None, then end_epoch can only be set to what its initial value was.
        if == -1, then end_epoch can be less than, equal, or greater than start_epoch.
        if == 0, then end_epoch can be equal to or greater than start_epoch.
        if == 1, then end_epoch can only be greater than start_epoch.
    :param kwargs: standard key word args, used to support multi inheritance
    """

    def __init__(
        self,
        start_epoch: float,
        min_start: float,
        end_epoch: float,
        min_end: float,
        end_comparator: Union[int, None],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._start_epoch = start_epoch
        self._init_start = start_epoch
        self._min_start = min_start
        self._end_epoch = end_epoch
        self._init_end = end_epoch
        self._min_end = min_end
        self._end_comparator = end_comparator
        self.validate_schedule()

    @ModifierProp()
    def start_epoch(self) -> float:
        """
        :return: The epoch to start the modifier at
            (set to -1.0 so it starts immediately)
        """
        return self._start_epoch

    @start_epoch.setter
    def start_epoch(self, value: float):
        """
        :param value: The epoch to start the modifier at
            (set to -1.0 so it starts immediately)
        """
        self._start_epoch = value
        self.validate_schedule()

    @ModifierProp()
    def end_epoch(self) -> float:
        """
        :return: The epoch to end the modifier at
            (set to -1.0 so it never ends)
        """
        return self._end_epoch

    @end_epoch.setter
    def end_epoch(self, value: float):
        """
        :param value: The epoch to end the modifier at (set to -1.0 so it never ends)
        """
        self._end_epoch = value
        self.validate_schedule()

    def validate_schedule(self):
        """
        Validate the schedule values of the params for the current instance are valid
        """

        if self._start_epoch < self._min_start:
            raise ValueError(
                "start_epoch of {} must be greater than or equal to {} for {}".format(
                    self._start_epoch, self._min_start, self.__class__.__name__
                )
            )

        if self._end_epoch < self._min_end:
            raise ValueError(
                "end_epoch of {} must be greater than or equal to {} for {}".format(
                    self._end_epoch, self._min_end, self.__class__.__name__
                )
            )

        if self._end_comparator is None and self._end_epoch != self._init_end:
            raise ValueError(
                "end_epoch of {} must be equal the init value of {} for {}".format(
                    self._end_epoch, self._init_end, self.__class__.__name__
                )
            )

        if self._end_comparator == 0 and self._start_epoch > self._end_epoch:
            raise ValueError(
                (
                    "end_epoch of {} must be greater than"
                    " or equal to start_epoch of {} for {}"
                ).format(self._end_epoch, self._start_epoch, self.__class__.__name__)
            )

        if self._end_comparator == 1 and self._start_epoch >= self._end_epoch:
            raise ValueError(
                "end_epoch of {} must be greater than start_epoch of {} for {}".format(
                    self._end_epoch, self._start_epoch, self.__class__.__name__
                )
            )


class BaseUpdate(BaseObject):
    """
    Abstract class meant to be used for all update modifiers.
    :py:func `~Modifier` and :py:func `~ScheduledModifier` are also meant
    to be inherited alongside this class.
    Handles base implementations for scheduled properties and methods
    to allow updates to be enforced.

    :param update_frequency: The number of epochs or fraction of epochs to
            update at between start and end
    :param min_frequency: The minimum acceptable value for update_frequency,
        default -1
    :param kwargs: standard key word args, used to support multi inheritance
    """

    def __init__(self, update_frequency: float, min_frequency: float, **kwargs):
        super().__init__(**kwargs)
        self._update_frequency = update_frequency
        self._min_frequency = min_frequency
        self.validate_update()

    @ModifierProp()
    def update_frequency(self) -> float:
        """
        :return: The number of epochs or fraction of epochs to update at between
            start and end
        """
        return self._update_frequency

    @update_frequency.setter
    def update_frequency(self, value: float):
        """
        :param value: The number of epochs or fraction of epochs to update at between
            start and end
        """
        self._update_frequency = value
        self.validate_update()

    def validate_update(self):
        """
        Validate the update schedule values of the params for the current instance
        are valid
        """

        if self._update_frequency < self._min_frequency:
            raise ValueError(
                (
                    "update_frequency of {} must be greater than or "
                    "equal to {} for {}"
                ).format(
                    self._update_frequency, self._min_frequency, self.__class__.__name__
                )
            )


class ModifierYAML(object):
    """
    A decorator to handle making a modifier class YAML ready.
    IE it can be loaded in through the yaml plugin easily.

    :param framework: the string representing the ML framework the modifier should
        be stored under
    """

    def __init__(self, framework: str):
        if not framework:
            raise ValueError("a framework is required")

        self._framework = framework

    def __call__(self, clazz):
        """
        :param clazz: the class to create yaml constructors for
        :return: the class after yaml constructors have been added
        """
        yaml_key = "{}".format(BaseModifier.yaml_key(clazz, self._framework))

        def constructor(loader, node):
            instance = clazz.__new__(clazz)
            yield instance
            state = loader.construct_mapping(node, deep=True)
            instance.__init__(**state)

        yaml.add_constructor(yaml_key, constructor)
        yaml.add_constructor(
            yaml_key,
            constructor,
            yaml.SafeLoader,
        )

        return clazz
