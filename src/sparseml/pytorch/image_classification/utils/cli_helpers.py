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
Callbacks and Utilities for CLI
"""

import json
import os
from typing import Any, Dict, Tuple

import click


__all__ = [
    "parse_json_callback",
    "create_dir_callback",
    "OptionEatAllArguments",
    "parse_into_tuple_of_ints",
    "parameters_to_dict",
]


def parse_json_callback(ctx, params, value: str) -> Dict:
    """
    Parse a json string into a dictionary

    :param ctx: The click context
    :param params: The click params
    :param value: The json string to parse
    :return: The parsed dictionary
    """
    # JSON string -> dict Callback
    if isinstance(value, str):
        return json.loads(value)
    return value


def create_dir_callback(ctx, params, value: str):
    """
    Create and return directory if it doesn't exist.

    :param ctx: The click context
    :param params: The click params
    :param value: The value to create the directory from
    :returns: The directory path
    """
    if value is None:
        return
    os.makedirs(value, exist_ok=True)
    return value


def parse_into_tuple_of_ints(ctx, params, value) -> Tuple[int, ...]:
    """
    Parse a string into a tuple of ints.

    :param ctx: The click context
    :param params: The click params
    :param value: The value to parse
    :return: Tuple of ints
    """
    if not value:
        return ()
    return tuple(int(element) for element in eval(value))


@click.pass_context
def parameters_to_dict(ctx) -> Dict[str, Any]:
    """
    Grab all the click parameters as a dict
    (where keys are parameter names and values are parameter values).

    :param ctx: The click context
    :return: Dictionary containing parameter names and values
    """
    parameters = ctx.params
    return parameters


class OptionEatAllArguments(click.Option):
    """
    A click.Option that eats all arguments. Click does not support nargs=-1 for
    options. This class is a work around for this limitation.
    Repurposed from https://stackoverflow.com/a/48394004
    """

    def __init__(self, *args, **kwargs):
        self.save_other_options = kwargs.pop("save_other_options", True)
        nargs = kwargs.pop("nargs", -1)
        assert nargs == -1, "nargs, if set, must be -1 not {}".format(nargs)
        super(OptionEatAllArguments, self).__init__(*args, **kwargs)
        self._previous_parser_process = None
        self._eat_all_parser = None

    def add_to_parser(self, parser, ctx):
        def parser_process(value, state):
            # method to hook to the parser.process
            done = False
            value = [value]
            if self.save_other_options:
                # grab everything up to the next option
                while state.rargs and not done:
                    for prefix in self._eat_all_parser.prefixes:
                        if state.rargs[0].startswith(prefix):
                            done = True
                    if not done:
                        value.append(state.rargs.pop(0))
            else:
                # grab everything remaining
                value += state.rargs
                state.rargs[:] = []
            value = tuple(value)

            # call the actual process
            self._previous_parser_process(value, state)

        retval = super(OptionEatAllArguments, self).add_to_parser(parser, ctx)
        for name in self.opts:
            our_parser = parser._long_opt.get(name) or parser._short_opt.get(name)
            if our_parser:
                self._eat_all_parser = our_parser
                self._previous_parser_process = our_parser.process
                our_parser.process = parser_process
                break

        return retval
