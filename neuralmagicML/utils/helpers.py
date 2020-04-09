"""
General utility helper functions.
Common functions for interfacing with python primitives and directories/files.
"""

from typing import Union, Iterable, Any
import sys
import os
import errno
import fnmatch


__all__ = [
    "ALL_TOKEN",
    "flatten_iterable",
    "convert_to_bool",
    "validate_str_iterable",
    "INTERPOLATION_FUNCS",
    "interpolate",
    "clean_path",
    "create_dirs",
    "create_parent_dirs",
    "create_unique_dir",
    "path_file_count",
]


ALL_TOKEN = "__ALL__"


##############################
#
# general python helper functions
#
##############################


def flatten_iterable(li: Iterable):
    """
    :param li: a possibly nested iterable of items to be flattened
    :return: a flattened version of the list where all elements are in a single list
             flattened in a depth first pattern
    """

    def _flatten_gen(_li):
        for el in _li:
            if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
                yield from _flatten_gen(el)
            else:
                yield el

    return list(_flatten_gen(li))


def convert_to_bool(val: Any):
    """
    :param val: the value to be converted to a bool,
        supports logical values as strings ie True, t, false, 0
    :return: the boolean representation of the value, if it can't be determined,
        falls back on returning True
    """
    return (
        bool(val)
        if not isinstance(val, str)
        else bool(val) and "f" not in val.lower() and "0" not in val.lower()
    )


def validate_str_iterable(
    val: Union[str, Iterable[str]], error_desc: str = ""
) -> Union[str, Iterable[str]]:
    """
    :param val: the value to validate, check that it is a list (and flattens it),
        otherwise checks that it's an ALL string, otherwise raises a ValueError
    :param error_desc: the description to raise an error with in the event that
        the val wasn't valid
    :return: the validated version of the param
    """
    if isinstance(val, str):
        if val.upper() != ALL_TOKEN:
            raise ValueError(
                "unsupported string ({}) given in {}".format(val, error_desc)
            )

        return val.upper()

    if isinstance(val, Iterable):
        return flatten_iterable(val)

    raise ValueError("unsupported type ({}) given in {}".format(val, error_desc))


INTERPOLATION_FUNCS = ["linear", "cubic", "inverse_cubic"]


def interpolate(
    x_cur: float, x0: float, x1: float, y0: Any, y1: Any, inter_func: str = "linear"
) -> Any:
    """
    note, caps values at their min of x0 and max x1,
    designed to not work outside of that range for implementation reasons

    :param x_cur: the current value for x, should be between x0 and x1
    :param x0: the minimum for x to interpolate between
    :param x1: the maximum for x to interpolate between
    :param y0: the minimum for y to interpolate between
    :param y1: the maximum for y to interpolate between
    :param inter_func: the type of function to interpolate with:
        linear, cubic, inverse_cubic
    :return: the interpolated value projecting x into y for the given
        interpolation function
    """
    if inter_func not in INTERPOLATION_FUNCS:
        raise ValueError(
            "unsupported inter_func given of {} must be one of {}".format(
                inter_func, INTERPOLATION_FUNCS
            )
        )

    # convert our x to 0-1 range since equations are designed to fit in
    # (0,0)-(1,1) space
    x_per = (x_cur - x0) / (x1 - x0)

    # map x to y using the desired function in (0,0)-(1,1) space
    if inter_func == "linear":
        y_per = x_per
    elif inter_func == "cubic":
        # https://www.wolframalpha.com/input/?i=1-(1-x)%5E3+from+0+to+1
        y_per = 1 - (1 - x_per) ** 3
    elif inter_func == "inverse_cubic":
        # https://www.wolframalpha.com/input/?i=1-(1-x)%5E(1%2F3)+from+0+to+1
        y_per = 1 - (1 - x_per) ** (1 / 3)
    else:
        raise ValueError(
            "unsupported inter_func given of {} in interpolate".format(inter_func)
        )

    if y_per <= 0.0 + sys.float_info.epsilon:
        return y0

    if y_per >= 1.0 - sys.float_info.epsilon:
        return y1

    # scale the threshold based on what we want the current to be
    return y_per * (y1 - y0) + y0


def clean_path(path: str) -> str:
    """
    :param path: the directory or file path to clean
    :return: a cleaned version that expands the user path and creates an absolute path
    """
    return os.path.abspath(os.path.expanduser(path))


def create_dirs(path: str):
    """
    :param path: the directory path to try and create
    """
    path = clean_path(path)

    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            # Unexpected OSError, re-raise.
            raise


def create_parent_dirs(path: str):
    """
    :param path: the file path to try to create the parent directories for
    """
    parent = os.path.dirname(path)
    create_dirs(parent)


def create_unique_dir(path: str, check_number: int = 0) -> str:
    """
    :param path: the file path to create a unique version of
        (append numbers until one doesn't exist)
    :param check_number: the number to begin checking for unique versions at
    :return: the unique directory path
    """
    check_path = clean_path("{}-{:04d}".format(path, check_number))

    if not os.path.exists(check_path):
        return check_path

    return create_unique_dir(path, check_number + 1)


def path_file_count(path: str, pattern: str = "*") -> int:
    """
    Return the number of files that match the given pattern under the given path

    :param path: the path to the directory to look for files under
    :param pattern: the pattern the files must match to be counted
    :return: the number of files matching the pattern under the directory
    """
    path = clean_path(path)

    return len(fnmatch.filter(os.listdir(path), pattern))
