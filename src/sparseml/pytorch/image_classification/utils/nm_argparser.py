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

import dataclasses
import json
import re
import sys
from argparse import ArgumentParser, ArgumentTypeError
from copy import copy
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, List, NewType, Optional, Tuple, Union


__all__ = [
    "NmArgumentParser",
    "string_to_bool",
]

DataClass = NewType("DataClass", Any)
DataClassType = NewType("DataClassType", Any)


# From https://stackoverflow.com/questions/15008758
# /parsing-boolean-values-with-argparse
def string_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError(
            f"Truthy value expected: got {v} but expected one of yes/no,"
            f"true/false, t/f, y/n, 1/0 (case insensitive)."
        )


# Inspired from https://huggingface.co/transformers/_modules
# /transformers/hf_argparser.html
class NmArgumentParser(ArgumentParser):
    """
    This subclass of `argparse.ArgumentParser` uses type hints on dataclasses
    to generate arguments.

    The class is designed to play well with the native argparse. In particular,
    you can add more (non-dataclass backed) arguments to the parser after
    initialization and you'll get the output back after parsing as an additional
    namespace. Optional: To create sub argument groups use the
    `_argument_group_name` attribute in the dataclass.

    Note: __post_init__(...) for specific dataclasses passed is executed only
    when parse_args_into_dataclasses(...) function is called because it needs
    actual instantiation of the dataclass.
    """

    dataclass_types: Iterable[DataClassType]

    def __init__(
        self, dataclass_types: Union[DataClassType, Iterable[DataClassType]], **kwargs
    ):
        """
        :param dataclass_types: Dataclass type, or list of dataclass types
            for which we will "fill" instances with the parsed args.
        :param kwargs: (Optional) Passed to `argparse.ArgumentParser()` in
            the regular way.
        """
        super().__init__(**kwargs)
        if dataclasses.is_dataclass(dataclass_types):
            dataclass_types = [dataclass_types]
        self.dataclass_types = dataclass_types
        for dataclass_ in self.dataclass_types:
            self._add_dataclass_arguments(dataclass_)

    def _add_dataclass_arguments(self, dataclass_: DataClassType):
        if hasattr(dataclass_, "_argument_group_name"):
            parser = self.add_argument_group(dataclass_._argument_group_name)
        else:
            parser = self
        for field in dataclasses.fields(dataclass_):
            if not field.init:
                continue

            name, kwargs = field.name, field.metadata.copy()

            keep_underscores_key = "keep_underscores"
            keep_underscores = kwargs.get(keep_underscores_key)
            _field_name = name if keep_underscores else name.replace("_", "-")

            # cleanup
            if keep_underscores_key in kwargs:
                del kwargs[keep_underscores_key]

            # field.metadata is not used at all by Data Classes,
            # it is provided as a third-party extension mechanism.
            if isinstance(field.type, str):
                raise ImportError(
                    "This implementation is not compatible with Postponed "
                    "Evaluation of Annotations (PEP 563),"
                    "which can be opted in from Python 3.7 with "
                    "`from __future__ import annotations`."
                )
            typestring = str(field.type)
            for prim_type in (int, float, str):
                for collection in (List,):
                    if (
                        typestring == f"typing.Union["
                        f"{collection[prim_type]}, NoneType]"
                        or typestring == f"typing.Optional" f"[{collection[prim_type]}]"
                    ):
                        field.type = collection[prim_type]
                if (
                    typestring == f"typing.Union[" f"{prim_type.__name__}, NoneType]"
                    or typestring == f"typing.Optional[" f"{prim_type.__name__}]"
                ):
                    field.type = prim_type

            if isinstance(field.type, type) and issubclass(field.type, Enum):
                kwargs["choices"] = [x.value for x in field.type]
                kwargs["type"] = type(kwargs["choices"][0])
                if field.default is not dataclasses.MISSING:
                    kwargs["default"] = field.default
                else:
                    kwargs["required"] = True
            elif field.type is bool or field.type == Optional[bool]:
                if field.default is True:
                    kwargs_copy = copy(kwargs)
                    if "help" in kwargs_copy:
                        kwargs_copy["help"] = f"Do not {kwargs_copy['help'].lower()}"
                    parser.add_argument(
                        f"--no-{_field_name}",
                        action="store_false",
                        dest=_field_name,
                        **kwargs_copy,
                    )

                # Hack because type=bool in argparse does not behave as we want.
                kwargs["type"] = string_to_bool
                if field.type is bool or (
                    field.default is not None
                    and field.default is not dataclasses.MISSING
                ):
                    # Default value is False if we have no default
                    # when of type bool.
                    if field.default is dataclasses.MISSING:
                        default = False
                    else:
                        default = field.default
                    # This is the value that will get picked if
                    # we don't include --field_name in any way
                    kwargs["default"] = default

                    # This tells argparse we accept 0 or 1
                    # value after --field_name
                    kwargs["nargs"] = "?"
                    # This is the value that will get picked
                    # if we do --field_name (without value)
                    kwargs["const"] = True
            elif (
                hasattr(field.type, "__origin__")
                and re.search(r"^typing\.List\[(.*)\]$", str(field.type)) is not None
            ):
                kwargs["nargs"] = "+"
                kwargs["type"] = field.type.__args__[0]
                assert all(
                    x == kwargs["type"] for x in field.type.__args__
                ), f"{field.name} cannot be a List of mixed types"
                if field.default_factory is not dataclasses.MISSING:
                    kwargs["default"] = field.default_factory()
                elif field.default is dataclasses.MISSING:
                    kwargs["required"] = True
            else:
                kwargs["type"] = field.type
                if field.default is not dataclasses.MISSING:
                    kwargs["default"] = field.default
                elif field.default_factory is not dataclasses.MISSING:
                    kwargs["default"] = field.default_factory()
                else:
                    kwargs["required"] = True
            parser.add_argument(f"--{_field_name}", **kwargs)

    def parse_args_into_dataclasses(
        self,
        args=None,
        return_remaining_strings=False,
        look_for_args_file=True,
        args_filename=None,
    ) -> Tuple[DataClass, ...]:
        """
        Parse command-line args into instances of the specified dataclass types.

        This relies on argparse's `ArgumentParser.parse_known_args`.
        See the doc at:
        docs.python.org/3.7/library/argparse.html#argparse.ArgumentParser.
        parse_args


        :param args: List of strings to parse. The default is taken from
            sys.argv. (same as argparse.ArgumentParser)
        :param return_remaining_strings: If true, also return a list of
            remaining argument strings.
        :param look_for_args_file: If true, will look for a ".args" file with
            the same base name as the entry point script for this process,
            and will append its potential content to the command line args.
        :param args_filename: If not None, will uses this file instead of the
            ".args" file specified in the previous argument.
        :returns: Tuple consisting of:
                    - the dataclass instances in the same order as they were
                    passed to the initializer.abspath
                    - if applicable, an additional namespace for more
                    (non-dataclass backed) arguments added to the parser
                    after initialization.
                    - The potential list of remaining argument strings.
                    (same as argparse.ArgumentParser.parse_known_args)
        """
        if args_filename or (look_for_args_file and len(sys.argv)):
            if args_filename:
                args_file = Path(args_filename)
            else:
                args_file = Path(sys.argv[0]).with_suffix(".args")

            if args_file.exists():
                fargs = args_file.read_text().split()
                args = fargs + args if args is not None else fargs + sys.argv[1:]
                # in case of duplicate arguments the first one has precedence
                # so we append rather than prepend.
        namespace, remaining_args = self.parse_known_args(args=args)
        outputs = []
        for dtype in self.dataclass_types:
            keys = {f.name for f in dataclasses.fields(dtype) if f.init}
            inputs = {k: v for k, v in vars(namespace).items() if k in keys}
            for k in keys:
                delattr(namespace, k)
            obj = dtype(**inputs)
            outputs.append(obj)
        if len(namespace.__dict__) > 0:
            # additional namespace.
            outputs.append(namespace)
        if return_remaining_strings:
            return (*outputs, remaining_args)
        else:
            if remaining_args:
                raise ValueError(
                    f"Some specified arguments are not used by the "
                    f"NmArgumentParser: {remaining_args}"
                )

            return (*outputs,)

    def parse_json_file(self, json_file: str) -> Tuple[DataClass, ...]:
        """
        Alternative helper method that does not use `argparse` at all,
        instead loading a json file and populating the dataclass types.
        """
        data = json.loads(Path(json_file).read_text())
        outputs = []
        for dtype in self.dataclass_types:
            keys = {f.name for f in dataclasses.fields(dtype) if f.init}
            inputs = {k: v for k, v in data.items() if k in keys}
            obj = dtype(**inputs)
            outputs.append(obj)
        return (*outputs,)

    def parse_dict(self, args: dict) -> Tuple[DataClass, ...]:
        """
        Alternative helper method that does not use `argparse` at all,
        instead uses a dict and populating the dataclass types.
        """
        outputs = []
        for dtype in self.dataclass_types:
            keys = {f.name for f in dataclasses.fields(dtype) if f.init}
            inputs = {k: v for k, v in args.items() if k in keys}
            obj = dtype(**inputs)
            outputs.append(obj)
        return (*outputs,)
