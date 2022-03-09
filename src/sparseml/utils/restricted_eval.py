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
Restricted eval function for safely evaluating equations in recipes
"""


import ast
import operator
from typing import Any, Dict, Optional


__all__ = [
    "restricted_eval",
    "UnknownVariableException",
]


class UnknownVariableException(Exception):
    """
    Exception raised for known variable names in restricted eval

    :param var_name: name of unknown variable
    """

    def __init__(self, var_name: str):
        self.var_name = var_name
        super().__init__(f"Unknown variable name in eval: {var_name}")


def restricted_eval(
    expression: str,
    variables: Optional[Dict[str, float]] = None,
) -> float:
    """
    :param expression: expression to evaluate
    :param variables: dictionary of string variables to float values that may be
        included in the expression
    :return: evaluated expression. Only supported operations, numbers, and float
        variables named in the variables dict may be included
    :raises: RuntimeError if any unsupported operations are included,
        UnknownVariableException if any variables not included in the variables dict
        are given
    """
    variables = variables or {}
    return _restricted_eval_node(ast.parse(expression.strip()).body[0], variables)


_VALID_BINOPS_TO_EVAL = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}

_VALID_UOPS_TO_EVAL = {ast.USub: operator.neg}

_VALID_FUNCTIONS_TO_EVAL = {
    "abs": abs,
    "float": float,
    "int": int,
    "min": min,
    "max": max,
    "round": round,
}


def _restricted_eval_node(node: Any, variables: Dict[str, float]) -> float:
    if isinstance(node, ast.Expr):
        return _restricted_eval_node(node.value, variables)
    if isinstance(node, ast.Num):
        return node.n
    if isinstance(node, ast.Name):
        if node.id in variables:
            return variables[node.id]
        else:
            raise UnknownVariableException(node.id)
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type in _VALID_BINOPS_TO_EVAL:
            return _VALID_BINOPS_TO_EVAL[op_type](
                _restricted_eval_node(node.left, variables),
                _restricted_eval_node(node.right, variables),
            )
        else:
            raise RuntimeError(f"Unsupported binary operator type {op_type}")
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type in _VALID_UOPS_TO_EVAL:
            return _VALID_UOPS_TO_EVAL[op_type](
                _restricted_eval_node(node.left, variables),
            )
        else:
            raise RuntimeError(f"Unsupported binary operator type {op_type}")
    if isinstance(node, ast.Call):
        func_name = node.func.id
        if func_name in _VALID_FUNCTIONS_TO_EVAL:
            args = [_restricted_eval_node(arg, variables) for arg in node.args]
            kwargs = {
                kwarg.arg: _restricted_eval_node(kwarg.value, variables)
                for kwarg in node.keywords
            }
            return _VALID_FUNCTIONS_TO_EVAL[func_name](*args, **kwargs)
        else:
            raise RuntimeError(f"Unsupported function name {func_name}")

    raise RuntimeError(f"Unsupported AST node type {type(node)}")
