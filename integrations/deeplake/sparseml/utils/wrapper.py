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
Code for properly merging function attributes for decorated / wrapped functions.
Merges docs, annotations, dicts, etc.
"""

from typing import Callable, List


__all__ = ["wrapper_decorator"]


def wrapper_decorator(wrapped: Callable):
    """
    A wrapper decorator to be applied as a decorator to a function.
    Merges the decorated function properties with wrapped.

    :param wrapped: the wrapped function to merge decorations with
    :return: the decorator to apply to the function
    """

    def decorator(wrapper: Callable):
        for attr in (
            "__module__",
            "__name__",
            "__qualname__",
        ):
            value = getattr(wrapped, attr)
            setattr(wrapper, attr, value)

        for attr in ("__dict__", "__annotations__"):
            getattr(wrapper, attr).update(getattr(wrapped, attr))

        _doc_merge(wrapped, wrapper)

        wrapper.__wrapped__ = wrapped

        return wrapper

    return decorator


def _get_doc_indent(lines: List[str]) -> str:
    for line in lines:
        if not line:
            continue

        leading_spaces = len(line) - len(line.lstrip())

        return "".join(" " for _ in range(leading_spaces))

    return ""


def _strip_doc_indent(doc: str) -> List[str]:
    if not doc:
        return []

    doc_lines = doc.splitlines()
    doc_indent = _get_doc_indent(doc_lines)
    doc_lines = [
        line if not line.startswith(doc_indent) else line[len(doc_indent) :]
        for line in doc_lines
    ]

    # remove empty lines at beginning and end to make merging cleaner
    while len(doc_lines) > 0 and not doc_lines[0]:
        doc_lines.pop(0)

    while len(doc_lines) > 0 and not doc_lines[-1]:
        doc_lines.pop(-1)

    return doc_lines


def _doc_merge(wrapped: Callable, wrapper: Callable):
    stripped_wrapped = _strip_doc_indent(wrapped.__doc__)
    stripped_wrapper = _strip_doc_indent(wrapper.__doc__)
    merge = []

    # check for return at end of doc string in wrapped
    if len(stripped_wrapped) > 0 and ":return" in stripped_wrapped[-1]:
        merge.extend(stripped_wrapped[:-1])
        merge.extend(stripped_wrapper)
        merge.append(stripped_wrapped[-1])
    else:
        merge.extend(stripped_wrapped)
        merge.extend(stripped_wrapper)

    wrapper.__doc__ = "\n".join(merge)
