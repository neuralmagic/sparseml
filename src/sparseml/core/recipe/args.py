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

import math
import re
from typing import Any, Dict, Optional, Union


__all__ = ["RecipeArgs"]


class RecipeArgs(Dict[str, Any]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._evaluated: "Optional[RecipeArgs]" = None

    def combine(self, other: Union["RecipeArgs", Dict[str, Any]]) -> "RecipeArgs":
        combined = RecipeArgs()
        combined.update(self)

        if other:
            combined.update(other)

        return combined

    def evaluate(self, parent: "RecipeArgs" = None) -> "RecipeArgs":
        self._evaluated = RecipeArgs.eval_args(self.combine(parent))

        return self._evaluated

    def evaluate_ext(self, target: Dict[str, Any]) -> Dict[str, Any]:
        args = RecipeArgs.eval_args(self)
        resolved = {}

        for key, value in target.items():
            resolved[key] = RecipeArgs.eval_obj(value, args)

        return resolved

    @staticmethod
    def eval_str(target: str, args: Dict[str, Any] = None) -> Union[str,float]:
        if "eval(" not in target:
            return target

        pattern = re.compile(r"eval\(([^()]*)\)")
        match = pattern.search(target)

        if not match:
            raise ValueError(f"invalid eval string {target}")

        inner_expr = match.group(1)
        result = eval(inner_expr, {"math": math}, args if args else {})
        new_target = target.replace(match.group(0), str(result))
        try:
            return float(new_target)
        except ValueError:
            return RecipeArgs.eval_str(new_target, args)

    @staticmethod
    def eval_args(args: Dict[str, Any]) -> "RecipeArgs":
        resolved = args.copy()

        while True:
            for key, value in resolved.items():
                if isinstance(value, str):
                    resolved[key] = RecipeArgs.eval_str(value, resolved)
                else:
                    resolved[key] = value

            if args == resolved:
                break
            else:
                args = resolved.copy()

        return RecipeArgs(resolved)

    @staticmethod
    def eval_obj(target: Any, args: Dict[str, Any] = None) -> Any:
        if isinstance(target, str):
            return RecipeArgs.eval_str(target, args)
        elif isinstance(target, dict):
            return {
                key: RecipeArgs.eval_obj(value, args) for key, value in target.items()
            }
        elif isinstance(target, list):
            return [RecipeArgs.eval_obj(item, args) for item in target]

        return target
