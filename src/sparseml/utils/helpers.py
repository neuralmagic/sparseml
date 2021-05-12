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
General utility helper functions.
Common functions for interfacing with python primitives and directories/files.
"""

import errno
import fnmatch
import logging
import os
import re
import sys
from collections import OrderedDict
from typing import Any, Callable, Dict, Iterable, List, Tuple, Union
from urllib.parse import urlparse

import numpy

from sparsezoo import Zoo
from sparsezoo.objects import Recipe
from sparsezoo.utils import load_numpy_list


__all__ = [
    "ALL_TOKEN",
    "ALL_PRUNABLE_TOKEN",
    "flatten_iterable",
    "convert_to_bool",
    "validate_str_iterable",
    "bucket_iterable",
    "INTERPOLATION_FUNCS",
    "interpolate",
    "interpolate_list_linear",
    "interpolated_integral",
    "clean_path",
    "create_dirs",
    "create_parent_dirs",
    "create_unique_dir",
    "path_file_count",
    "path_file_size",
    "is_url",
    "NDARRAY_KEY",
    "load_numpy",
    "save_numpy",
    "load_labeled_data",
    "NumpyArrayBatcher",
    "tensor_export",
    "tensors_export",
    "load_recipe_yaml_str",
    "parse_optimization_str",
]


ALL_TOKEN = "__ALL__"
ALL_PRUNABLE_TOKEN = "__ALL_PRUNABLE__"
_LOGGER = logging.getLogger(__name__)


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
        otherwise checks that it's an __ALL__ or __ALL_PRUNABLE__ string,
        otherwise raises a ValueError
    :param error_desc: the description to raise an error with in the event that
        the val wasn't valid
    :return: the validated version of the param
    """
    if isinstance(val, str):
        if val.upper() != ALL_TOKEN and val.upper() != ALL_PRUNABLE_TOKEN:
            raise ValueError(
                "unsupported string ({}) given in {}".format(val, error_desc)
            )

        return val.upper()

    if isinstance(val, Iterable):
        return flatten_iterable(val)

    raise ValueError("unsupported type ({}) given in {}".format(val, error_desc))


def bucket_iterable(
    val: Iterable[Any],
    num_buckets: int = 3,
    edge_percent: float = 0.05,
    sort_highest: bool = True,
    sort_key: Callable[[Any], Any] = None,
) -> List[Tuple[int, Any]]:
    """
    Bucket iterable into subarray consisting of the first top percentage
    followed by the rest of the iterable sliced into equal sliced groups.

    :param val: The iterable to bucket
    :param num_buckets: The number of buckets to group the iterable into,
        does not include the top bucket
    :param edge_percent: Group the first percent into its own bucket.
        If sort_highest, then this is the top percent, else bottom percent.
        If <= 0, then will not create an edge bucket
    :param sort_highest: True to sort such that the highest percent is first
        and will create buckets in descending order.
        False to sort so lowest is first and create buckets in ascending order.
    :param sort_key: The sort_key, if any, to use for sorting the iterable
        after converting it to a list
    :return: a list of each value mapped to the bucket it was sorted into
    """

    val_list = [v for v in val]
    val_list.sort(key=sort_key, reverse=sort_highest)
    bucketed_values = []
    edge_count = round(edge_percent * len(val_list))

    if edge_count > 0:
        bucketed_values.extend([(-1, val) for val in val_list[:edge_count]])
        val_list = val_list[edge_count:]

    buckets_count = round(len(val_list) / float(num_buckets))

    for bucket in range(num_buckets):
        add_vals = val_list[:buckets_count] if bucket < num_buckets - 1 else val_list
        val_list = val_list[buckets_count:] if bucket < num_buckets - 1 else []
        bucketed_values.extend([(bucket, val) for val in add_vals])

    return bucketed_values


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


def interpolate_list_linear(
    measurements: List[Tuple[float, float]], x_val: Union[float, List[float]]
) -> List[Tuple[float, float]]:
    """
    interpolate for input values within a list of measurements linearly

    :param measurements: the measurements to interpolate the output value between
    :param x_val: the target values to interpolate to the second dimension
    :return: a list of tuples containing the target values, interpolated values
    """
    assert len(measurements) > 1
    measurements.sort(key=lambda v: v[0])

    x_vals = [x_val] if isinstance(x_val, float) else x_val
    x_vals.sort()

    interpolated = []
    lower_index = 0
    higher_index = 1

    for x_val in x_vals:
        while (
            x_val > measurements[higher_index][0]
            and higher_index < len(measurements) - 1
        ):
            lower_index += 1
            higher_index += 1

        x0, y0 = measurements[lower_index]
        x1, y1 = measurements[higher_index]
        y_val = y0 + (x_val - x0) * ((y1 - y0) / (x1 - x0))
        interpolated.append((x_val, y_val))

    return interpolated


def interpolated_integral(measurements: List[Tuple[float, float]]):
    """
    Calculate the interpolated integal for a group of measurements of the form
    [(x0, y0), (x1, y1), ...]

    :param measurements: the measurements to calculate the integral for
    :return: the integral or area under the curve for the measurements given
    """
    if len(measurements) < 1:
        return 0.0

    if len(measurements) == 1:
        return measurements[0][1]

    measurements.sort(key=lambda v: v[0])
    integral = 0.0

    for index, (x_val, y_val) in enumerate(measurements):
        if index >= len(measurements) - 1:
            continue

        x_next, y_next = measurements[index + 1]
        x_dist = x_next - x_val
        area = y_val * x_dist + (y_next - y_val) * x_dist / 2.0
        integral += area

    return integral


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


def path_file_size(path: str) -> int:
    """
    Return the total size, in bytes, for a path on the file system

    :param path: the path (directory or file) to get the size for
    :return: the size of the path, in bytes, as stored on disk
    """

    if not os.path.isdir(path):
        stat = os.stat(path)

        return stat.st_size

    total_size = 0
    seen = {}

    for dir_path, dir_names, filenames in os.walk(path):
        for file in filenames:
            file_path = os.path.join(dir_path, file)

            try:
                stat = os.stat(file_path)
            except OSError:
                continue

            try:
                seen[stat.st_ino]
            except KeyError:
                seen[stat.st_ino] = True
            else:
                continue

            total_size += stat.st_size

    return total_size


def is_url(val: str):
    """
    :param val: value to check if it is a url or not
    :return: True if value is a URL, False otherwise
    """

    try:
        result = urlparse(val)

        return all([result.scheme, result.netloc])
    except ValueError:
        return False


##############################
#
# numpy helper functions
#
##############################


NDARRAY_KEY = "ndarray"


def load_numpy(file_path: str) -> Union[numpy.ndarray, Dict[str, numpy.ndarray]]:
    """
    Load a numpy file into either an ndarray or an OrderedDict representing what
    was in the npz file

    :param file_path: the file_path to load
    :return: the loaded values from the file
    """
    file_path = clean_path(file_path)
    array = numpy.load(file_path)

    if not isinstance(array, numpy.ndarray):
        tmp_arrray = array
        array = OrderedDict()
        for key, val in tmp_arrray.items():
            array[key] = val

    return array


def save_numpy(
    array: Union[numpy.ndarray, Dict[str, numpy.ndarray], Iterable[numpy.ndarray]],
    export_dir: str,
    name: str,
    npz: bool = True,
):
    """
    Save a numpy array or collection of numpy arrays to disk

    :param array: the array or collection of arrays to save
    :param export_dir: the directory to export the numpy file into
    :param name: the name of the file to export to (without extension)
    :param npz: True to save as an npz compressed file, False for standard npy.
        Note, npy can only be used for single numpy arrays
    :return: the saved path
    """
    create_dirs(export_dir)
    export_path = os.path.join(
        export_dir, "{}.{}".format(name, "npz" if npz else "npy")
    )

    if isinstance(array, numpy.ndarray) and npz:
        numpy.savez_compressed(export_path, array)
    elif isinstance(array, numpy.ndarray):
        numpy.save(export_path, array)
    elif isinstance(array, Dict) and npz:
        numpy.savez_compressed(export_path, **array)
    elif isinstance(array, Dict):
        raise ValueError("Dict can only be exported to an npz file")
    elif isinstance(array, Iterable) and npz:
        numpy.savez_compressed(export_path, *[val for val in array])
    elif isinstance(array, Iterable):
        raise ValueError("Iterable can only be exported to an npz file")
    else:
        raise ValueError("Unrecognized type given for array {}".format(array))

    return export_path


def load_labeled_data(
    data: Union[str, Iterable[Union[str, numpy.ndarray, Dict[str, numpy.ndarray]]]],
    labels: Union[
        None, str, Iterable[Union[str, numpy.ndarray, Dict[str, numpy.ndarray]]]
    ],
    raise_on_error: bool = True,
) -> List[
    Tuple[
        Union[numpy.ndarray, Dict[str, numpy.ndarray]],
        Union[None, numpy.ndarray, Dict[str, numpy.ndarray]],
    ]
]:
    """
    Load labels and data from disk or from memory and group them together.
    Assumes sorted ordering for on disk. Will match between when a file glob is passed
    for either data and/or labels.

    :param data: the file glob, file path to numpy data tar ball, or list of arrays to
        use for data
    :param labels: the file glob, file path to numpy data tar ball, or list of arrays
        to use for labels, if any
    :param raise_on_error: True to raise on any error that occurs;
        False to log a warning, ignore, and continue
    :return: a list containing tuples of the data, labels. If labels was passed in
        as None, will now contain a None for the second index in each tuple
    """
    if isinstance(data, str):
        data = load_numpy_list(data)

    if labels is None:
        labels = [None for _ in range(len(data))]
    elif isinstance(labels, str):
        labels = load_numpy_list(labels)

    if len(data) != len(labels) and labels:
        # always raise this error, lengths must match
        raise ValueError(
            "len(data) given of {} does not match len(labels) given of {}".format(
                len(data), len(labels)
            )
        )

    labeled_data = []

    for dat, lab in zip(data, labels):
        try:
            if isinstance(dat, str):
                dat = load_numpy(dat)

            if lab is not None and isinstance(lab, str):
                lab = load_numpy(lab)

            labeled_data.append((dat, lab))
        except Exception as err:
            if raise_on_error:
                raise err
            else:
                _LOGGER.error("Error creating labeled data: {}".format(err))

    return labeled_data


class NumpyArrayBatcher(object):
    """
    Batcher instance to handle taking in dictionaries of numpy arrays,
    appending multiple items to them to increase their batch size,
    and then stack them into a single batched numpy array for all keys in the dicts.
    """

    def __init__(self):
        self._items = OrderedDict()  # type: Dict[str, List[numpy.ndarray]]

    def __len__(self):
        if len(self._items) == 0:
            return 0

        return len(self._items[list(self._items.keys())[0]])

    def append(self, item: Union[numpy.ndarray, Dict[str, numpy.ndarray]]):
        """
        Append a new item into the current batch.
        All keys and shapes must match the current state.

        :param item: the item to add for batching
        """
        if len(self) < 1 and isinstance(item, numpy.ndarray):
            self._items[NDARRAY_KEY] = [item]
        elif len(self) < 1:
            for key, val in item.items():
                self._items[key] = [val]
        elif isinstance(item, numpy.ndarray):
            if NDARRAY_KEY not in self._items:
                raise ValueError(
                    "numpy ndarray passed for item, but prev_batch does not contain one"
                )

            if item.shape != self._items[NDARRAY_KEY][0].shape:
                raise ValueError(
                    (
                        "item of numpy ndarray of shape {} does not "
                        "match the current batch shape of {}".format(
                            item.shape, self._items[NDARRAY_KEY][0].shape
                        )
                    )
                )

            self._items[NDARRAY_KEY].append(item)
        else:
            diff_keys = list(set(item.keys()) - set(self._items.keys()))

            if len(diff_keys) > 0:
                raise ValueError(
                    (
                        "numpy dict passed for item, not all keys match "
                        "with the prev_batch. difference: {}"
                    ).format(diff_keys)
                )

            for key, val in item.items():
                if val.shape != self._items[key][0].shape:
                    raise ValueError(
                        (
                            "item with key {} of shape {} does not "
                            "match the current batch shape of {}".format(
                                key, val.shape, self._items[key][0].shape
                            )
                        )
                    )

                self._items[key].append(val)

    def stack(self) -> Dict[str, numpy.ndarray]:
        """
        Stack the current items into a batch along a new, zeroed dimension

        :return: the stacked items
        """
        batch_dict = OrderedDict()

        for key, val in self._items.items():
            batch_dict[key] = numpy.stack(self._items[key])

        return batch_dict


def tensor_export(
    tensor: Union[numpy.ndarray, Dict[str, numpy.ndarray], Iterable[numpy.ndarray]],
    export_dir: str,
    name: str,
    npz: bool = True,
) -> str:
    """
    :param tensor: tensor to export to a saved numpy array file
    :param export_dir: the directory to export the file in
    :param name: the name of the file, .npy will be appended to it
    :param npz: True to export as an npz file, False otherwise
    :return: the path of the numpy file the tensor was exported to
    """
    create_dirs(export_dir)
    export_path = os.path.join(
        export_dir, "{}.{}".format(name, "npz" if npz else "npy")
    )

    if isinstance(tensor, numpy.ndarray) and npz:
        numpy.savez_compressed(export_path, tensor)
    elif isinstance(tensor, numpy.ndarray):
        numpy.save(export_path, tensor)
    elif isinstance(tensor, Dict) and npz:
        numpy.savez_compressed(export_path, **tensor)
    elif isinstance(tensor, Dict):
        raise ValueError("tensor dictionaries can only be saved as npz")
    elif isinstance(tensor, Iterable) and npz:
        numpy.savez_compressed(export_path, *tensor)
    elif isinstance(tensor, Iterable):
        raise ValueError("tensor iterables can only be saved as npz")
    else:
        raise ValueError("unknown type give for tensor {}".format(tensor))

    return export_path


def tensors_export(
    tensors: Union[numpy.ndarray, Dict[str, numpy.ndarray], Iterable[numpy.ndarray]],
    export_dir: str,
    name_prefix: str,
    counter: int = 0,
    break_batch: bool = False,
) -> List[str]:
    """
    :param tensors: the tensors to export to a saved numpy array file
    :param export_dir: the directory to export the files in
    :param name_prefix: the prefix name for the tensors to save as, will append
        info about the position of the tensor in a list or dict in addition
        to the .npy file format
    :param counter: the current counter to save the tensor at
    :param break_batch: treat the tensor as a batch and break apart into
        multiple tensors
    :return: the exported paths
    """
    create_dirs(export_dir)
    exported_paths = []

    if break_batch:
        _tensors_export_batch(tensors, export_dir, name_prefix, counter, exported_paths)
    else:
        _tensors_export_recursive(
            tensors, export_dir, name_prefix, counter, exported_paths
        )

    return exported_paths


def _tensors_export_recursive(
    tensors: Union[numpy.ndarray, Iterable[numpy.ndarray]],
    export_dir: str,
    name_prefix: str,
    counter: int,
    exported_paths: List[str],
):
    if isinstance(tensors, numpy.ndarray):
        exported_paths.append(
            tensor_export(tensors, export_dir, "{}-{:04d}".format(name_prefix, counter))
        )

        return

    if isinstance(tensors, Dict):
        raise ValueError("tensors dictionary is not supported for non batch export")

    if isinstance(tensors, Iterable):
        for index, tens in enumerate(tensors):
            _tensors_export_recursive(
                tens,
                export_dir,
                name_prefix,
                counter + index,
                exported_paths,
            )

        return

    raise ValueError(
        "unrecognized type for tensors given of {}".format(tensors.__class__.__name__)
    )


def _tensors_export_batch(
    tensors: Union[numpy.ndarray, Dict[str, numpy.ndarray], Iterable[numpy.ndarray]],
    export_dir: str,
    name_prefix: str,
    counter: int,
    exported_paths: List[str],
):
    if isinstance(tensors, numpy.ndarray):
        for index, tens in enumerate(tensors):
            exported_paths.append(
                tensor_export(
                    tens, export_dir, "{}-{:04d}".format(name_prefix, counter + index)
                )
            )

        return

    if isinstance(tensors, Dict):
        tensors = OrderedDict([(key, val) for key, val in tensors.items()])
        keys = [key for key in tensors.keys()]

        for index, tens in enumerate(zip(*tensors.values())):
            tens = OrderedDict([(key, val) for key, val in zip(keys, tens)])
            exported_paths.append(
                tensor_export(
                    tens, export_dir, "{}-{:04d}".format(name_prefix, counter + index)
                )
            )

        return

    if isinstance(tensors, Iterable):
        for index, tens in enumerate(zip(*tensors)):
            exported_paths.append(
                tensor_export(
                    tens, export_dir, "{}-{:04d}".format(name_prefix, counter + index)
                )
            )

        return

    raise ValueError(
        "unrecognized type for tensors given of {}".format(tensors.__class__.__name__)
    )


def load_recipe_yaml_str(file_path: Union[str, Recipe]) -> str:
    """
    Loads a YAML recipe file to a string or
    extracts recipe from YAML front matter in a sparsezoo markdown recipe card.
    Recipes can also be provided as SparseZoo model stubs or Recipe
    objects.

    YAML front matter: https://jekyllrb.com/docs/front-matter/

    :param file_path: file path to recipe YAML file or markdown recipe card or
        stub to a SparseZoo model whose recipe will be downloaded and loaded.
        SparseZoo stubs should be preceded by 'zoo:', and can contain an optional
        '?recipe_type=<type>' parameter or include a `/<type>` subpath. Can also
        be a SparseZoo Recipe object. i.e. '/path/to/local/recipe.yaml',
        'zoo:model/stub/path', 'zoo:model/stub/path?recipe_type=transfer_learn',
        'zoo:model/stub/path/transfer_learn'
    :return: the recipe YAML configuration loaded as a string
    """
    if isinstance(file_path, Recipe):
        # download and unwrap Recipe object
        file_path = file_path.downloaded_path()

    if not isinstance(file_path, str):
        raise ValueError(f"file_path must be a str, given {type(file_path)}")

    if file_path.startswith("zoo:"):
        # download from zoo stub
        recipe = Zoo.download_recipe_from_stub(file_path)
        file_path = recipe.downloaded_path()

    # load the yaml string
    if "\n" in file_path or "\r" in file_path:
        # treat as raw yaml passed in
        yaml_str = file_path
        extension = "unknown"
    else:
        # load yaml from file_path
        extension = file_path.lower().split(".")[-1]
        if extension not in ["md", "yaml"]:
            raise ValueError(
                "Unsupported file extension for recipe. Excepted '.md' or '.yaml'. "
                "Received {}".format(file_path)
            )
        with open(file_path, "r") as yaml_file:
            yaml_str = yaml_file.read()

    if extension == "md" or extension == "unknown":
        # extract YAML front matter from markdown recipe card
        # adapted from
        # https://github.com/jonbeebe/frontmatter/blob/master/frontmatter
        yaml_delim = r"(?:---|\+\+\+)"
        yaml = r"(.*?)"
        re_pattern = r"^\s*" + yaml_delim + yaml + yaml_delim
        regex = re.compile(re_pattern, re.S | re.M)
        result = regex.search(yaml_str)

        if result:
            yaml_str = result.group(1)
        elif extension == "md":
            # fail if we know whe should have extracted front matter out
            raise RuntimeError(
                "Could not extract YAML front matter from recipe card:"
                " {}".format(file_path)
            )

    return yaml_str


def parse_optimization_str(optim_full_name: str) -> Tuple[str, str, Any]:
    """
    :param optim_full_name: A name of a pretrained model optimization. i.e.
        'pruned-moderate-deepsparse', 'pruned-aggressive', 'base'
    :return: A tuple representing the corresponding SparseZoo model sparse_name,
        sparse_category, and sparse_target values with appropriate defaults when
        not present.
    """
    optim_defaults = ["base", "none", None]
    optim_split_name = optim_full_name.split("-")
    while len(optim_split_name) < len(optim_defaults):
        optim_split_name.append(optim_defaults[len(optim_split_name)])
    sparse_name, sparse_category, sparse_target = optim_split_name[:3]
    return sparse_name, sparse_category, sparse_target
