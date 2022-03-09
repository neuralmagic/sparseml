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

import argparse
import unittest
from argparse import Namespace
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from nm_argparser_ import NmArgumentParser, string_to_bool


def list_field(default=None, metadata=None):
    return field(default_factory=lambda: default, metadata=metadata)


@dataclass
class BasicExample:
    foo: int
    bar: float
    baz: str
    flag: bool


@dataclass
class WithDefaultExample:
    foo: int = 42
    baz: str = field(default="toto", metadata={"help": "help message"})


@dataclass
class WithDefaultBoolExample:
    foo: bool = False
    baz: bool = True
    opt: Optional[bool] = None


class BasicEnum(Enum):
    titi = "titi"
    toto = "toto"


@dataclass
class EnumExample:
    foo: BasicEnum = "toto"

    def __post_init__(self):
        self.foo = BasicEnum(self.foo)


@dataclass
class OptionalExample:
    foo: Optional[int] = None
    bar: Optional[float] = field(default=None, metadata={"help": "help message"})
    baz: Optional[str] = None
    ces: Optional[List[str]] = list_field(default=[])
    des: Optional[List[int]] = list_field(default=[])


@dataclass
class ListExample:
    foo_int: List[int] = list_field(default=[])
    bar_int: List[int] = list_field(default=[1, 2, 3])
    foo_str: List[str] = list_field(default=["Hallo", "Bonjour", "Hello"])
    foo_float: List[float] = list_field(default=[0.1, 0.2, 0.3])


@dataclass
class RequiredExample:
    required_list: List[int] = field()
    required_str: str = field()
    required_enum: BasicEnum = field()

    def __post_init__(self):
        self.required_enum = BasicEnum(self.required_enum)


@dataclass
class NonHyphenatedExample:
    non_hyphenated: int = field(
        metadata={
            "keep_underscores": True,
        }
    )


class NmArgumentParserTest(unittest.TestCase):
    def argparsersEqual(
        self, parser: argparse.ArgumentParser, expected_parser: argparse.ArgumentParser
    ) -> bool:
        """
        Small helper to check pseudo-equality of parsed arguments on
        `ArgumentParser` instances.
        """
        self.assertEqual(len(parser._actions), len(expected_parser._actions))
        for x, y in zip(parser._actions, expected_parser._actions):
            xx = {k: v for k, v in vars(x).items() if k != "container"}
            yy = {k: v for k, v in vars(y).items() if k != "container"}
            self.assertEqual(xx, yy)

    def test_basic(self):
        parser = NmArgumentParser(BasicExample)

        expected = argparse.ArgumentParser()
        expected.add_argument("--foo", type=int, required=True)
        expected.add_argument("--bar", type=float, required=True)
        expected.add_argument("--baz", type=str, required=True)
        expected.add_argument(
            "--flag", type=string_to_bool, default=False, const=True, nargs="?"
        )
        self.argparsersEqual(parser, expected)

        args = ["--foo", "1", "--baz", "quux", "--bar", "0.5"]
        (example,) = parser.parse_args_into_dataclasses(args, look_for_args_file=False)
        self.assertFalse(example.flag)

    def test_with_default(self):
        parser = NmArgumentParser(WithDefaultExample)

        expected = argparse.ArgumentParser()
        expected.add_argument("--foo", default=42, type=int)
        expected.add_argument("--baz", default="toto", type=str, help="help message")
        self.argparsersEqual(parser, expected)

    def test_with_default_bool(self):
        parser = NmArgumentParser(WithDefaultBoolExample)

        expected = argparse.ArgumentParser()
        expected.add_argument(
            "--foo", type=string_to_bool, default=False, const=True, nargs="?"
        )
        expected.add_argument("--no-baz", action="store_false", dest="baz")
        expected.add_argument(
            "--baz", type=string_to_bool, default=True, const=True, nargs="?"
        )
        expected.add_argument("--opt", type=string_to_bool, default=None)
        self.argparsersEqual(parser, expected)

        args = parser.parse_args([])
        self.assertEqual(args, Namespace(foo=False, baz=True, opt=None))

        args = parser.parse_args(["--foo", "--no-baz"])
        self.assertEqual(args, Namespace(foo=True, baz=False, opt=None))

        args = parser.parse_args(["--foo", "--baz"])
        self.assertEqual(args, Namespace(foo=True, baz=True, opt=None))

        args = parser.parse_args(["--foo", "True", "--baz", "True", "--opt", "True"])
        self.assertEqual(args, Namespace(foo=True, baz=True, opt=True))

        args = parser.parse_args(["--foo", "False", "--baz", "False", "--opt", "False"])
        self.assertEqual(args, Namespace(foo=False, baz=False, opt=False))

    def test_with_enum(self):
        parser = NmArgumentParser(EnumExample)

        expected = argparse.ArgumentParser()
        expected.add_argument(
            "--foo", default="toto", choices=["titi", "toto"], type=str
        )
        self.argparsersEqual(parser, expected)

        args = parser.parse_args([])
        self.assertEqual(args.foo, "toto")
        enum_ex = parser.parse_args_into_dataclasses([])[0]
        self.assertEqual(enum_ex.foo, BasicEnum.toto)

        args = parser.parse_args(["--foo", "titi"])
        self.assertEqual(args.foo, "titi")
        enum_ex = parser.parse_args_into_dataclasses(["--foo", "titi"])[0]
        self.assertEqual(enum_ex.foo, BasicEnum.titi)

    def test_with_list(self):
        parser = NmArgumentParser(ListExample)

        expected = argparse.ArgumentParser()
        expected.add_argument("--foo-int", nargs="+", default=[], type=int)
        expected.add_argument("--bar-int", nargs="+", default=[1, 2, 3], type=int)
        expected.add_argument(
            "--foo-str", nargs="+", default=["Hallo", "Bonjour", "Hello"], type=str
        )
        expected.add_argument(
            "--foo-float", nargs="+", default=[0.1, 0.2, 0.3], type=float
        )

        self.argparsersEqual(parser, expected)

        args = parser.parse_args([])
        self.assertEqual(
            args,
            Namespace(
                foo_int=[],
                bar_int=[1, 2, 3],
                foo_str=["Hallo", "Bonjour", "Hello"],
                foo_float=[0.1, 0.2, 0.3],
            ),
        )

        args = parser.parse_args(
            "--foo-int 1 --bar-int 2 3 --foo-str parser expected_parser "
            "c --foo-float 0.1 0.7".split()
        )
        self.assertEqual(
            args,
            Namespace(
                foo_int=[1],
                bar_int=[2, 3],
                foo_str=["parser", "expected_parser", "c"],
                foo_float=[0.1, 0.7],
            ),
        )

    def test_with_optional(self):
        parser = NmArgumentParser(OptionalExample)

        expected = argparse.ArgumentParser()
        expected.add_argument("--foo", default=None, type=int)
        expected.add_argument("--bar", default=None, type=float, help="help message")
        expected.add_argument("--baz", default=None, type=str)
        expected.add_argument("--ces", nargs="+", default=[], type=str)
        expected.add_argument("--des", nargs="+", default=[], type=int)
        self.argparsersEqual(parser, expected)

        args = parser.parse_args([])
        self.assertEqual(args, Namespace(foo=None, bar=None, baz=None, ces=[], des=[]))

        args = parser.parse_args(
            "--foo 12 --bar 3.14 --baz 42 --ces parser expected_parser "
            "c --des 1 2 3".split()
        )
        self.assertEqual(
            args,
            Namespace(
                foo=12,
                bar=3.14,
                baz="42",
                ces=["parser", "expected_parser", "c"],
                des=[1, 2, 3],
            ),
        )

    def test_with_required(self):
        parser = NmArgumentParser(RequiredExample)

        expected = argparse.ArgumentParser()
        expected.add_argument("--required-list", nargs="+", type=int, required=True)
        expected.add_argument("--required-str", type=str, required=True)
        expected.add_argument(
            "--required-enum", type=str, choices=["titi", "toto"], required=True
        )
        self.argparsersEqual(parser, expected)

    def test_parse_dict(self):
        parser = NmArgumentParser(BasicExample)

        args_dict = {
            "foo": 12,
            "bar": 3.14,
            "baz": "42",
            "flag": True,
        }

        parsed_args = parser.parse_dict(args_dict)[0]
        args = BasicExample(**args_dict)
        self.assertEqual(parsed_args, args)

    def test_non_hyphenated(self):
        parser = NmArgumentParser(dataclass_types=NonHyphenatedExample)
        args = parser.parse_args("--non_hyphenated 1".split())

        self.assertEqual(args, Namespace(non_hyphenated=1))
