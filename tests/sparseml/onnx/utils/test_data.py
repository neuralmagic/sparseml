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
import os
import tempfile
from typing import Dict, NamedTuple, Tuple, Union

import numpy
import pytest

from sparseml.onnx.utils import DataLoader
from sparsezoo import Model


DataloaderModelFixture = NamedTuple(
    "DataloaderModelFixture",
    [
        ("model_path", str),
        ("data_shape", Dict[str, Tuple[int, ...]]),
        ("label_shape", Union[None, Dict[str, Tuple[int, ...]]]),
        ("data_types", numpy.dtype),
    ],
)


@pytest.fixture(
    params=[
        (
            "zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/base-none",
            {"input": (1, 3, 224, 224)},
            {"output_0": (1, 1000), "output_1": (1, 1000)},
            {"input": numpy.dtype("float32")},
        ),
        (
            "zoo:cv/classification/mobilenet_v1-1.0/pytorch/sparseml/imagenet/base-none",  # noqa 501
            {"input": (1, 3, 224, 224)},
            {"output_0": (1, 1000)},
            {"input": numpy.dtype("float32")},
        ),
    ]
)
def dataloader_models(request) -> DataloaderModelFixture:
    model_stub, input_shapes, output_shapes, data_types = request.param
    model = Model(model_stub)
    model_path = model.onnx_model.path

    return DataloaderModelFixture(model_path, input_shapes, output_shapes, data_types)


def _test_dataloader(
    dataloader: DataLoader,
    data_shapes: Dict[str, Tuple[int, ...]],
    label_shapes: Union[None, Dict[str, Tuple[int, ...]]],
    batch_size: int,
    iter_steps: int,
    num_samples: int,
    data_types: Dict[str, numpy.dtype] = None,
):
    assert dataloader.batch_size == batch_size
    assert dataloader.iter_steps == iter_steps
    assert dataloader.infinite == (iter_steps == -1)
    if dataloader.iter_steps > 0:
        assert len(dataloader) == iter_steps
    elif dataloader.iter_steps < 0:
        assert len(dataloader) == 0
    else:
        assert len(dataloader) == math.ceil(num_samples / float(batch_size))

    iterations = 0
    for data, label in dataloader:
        if dataloader.infinite and iterations == iter_steps + 5:
            break
        for key in data_shapes:
            if data_types is not None and key in data_types:
                assert data[key].dtype == data_types[key]
            assert data[key].shape == (batch_size,) + data_shapes[key]

        if label_shapes is None:
            assert label is None
        else:
            for key in label_shapes:
                assert label[key].shape == (batch_size,) + label_shapes[key]
        iterations += 1
    assert (dataloader.infinite and iterations == iter_steps + 5) or (
        iterations == len(dataloader)
    )


@pytest.mark.parametrize(
    "data_shapes,label_shapes,batch_size,iter_steps,num_samples,data_types",
    [
        ({"0000": (3, 16, 16)}, None, 3, 1, 30, None),
        ({"0000": (3, 16, 16)}, None, 3, 2, 30, None),
        ({"0000": (3, 16, 16)}, None, 3, 2, 30, {"0000": numpy.dtype("int")}),
        (
            {"0000": (3, 16, 16), "0001": (4, 20, 20)},
            None,
            3,
            2,
            30,
            {"0000": numpy.int64, "0001": numpy.float32},
        ),
        ({"0000": (3, 16, 16)}, {"0000": (1000, 1)}, 3, 20, 30, None),
        (
            {"0000": (3, 16, 16), "0001": (4, 20, 20)},
            {"0000": (1000, 1), "0001": (1,)},
            3,
            20,
            30,
            None,
        ),
        ({"0000": (3, 16, 16)}, None, 3, 0, 30, None),
        ({"0000": (3, 16, 16)}, None, 3, -1, 30, None),
    ],
)
def test_dataloader_from_random(
    data_shapes: Dict[str, Tuple[int, ...]],
    label_shapes: Union[None, Dict[str, Tuple[int, ...]]],
    batch_size: int,
    iter_steps: int,
    num_samples: int,
    data_types: Dict[str, numpy.dtype],
):
    dataloader = DataLoader.from_random(
        data_shapes, label_shapes, batch_size, iter_steps, num_samples, data_types
    )
    _test_dataloader(
        dataloader,
        data_shapes,
        label_shapes,
        batch_size,
        iter_steps,
        num_samples,
        data_types,
    )


@pytest.mark.parametrize(
    "batch_size,iter_steps,num_samples,create_labels,strip_first_dim",
    [
        (10, 0, 100, False, True),
        (10, 0, 98, False, True),
        (10, -1, 100, False, True),
        (10, 10, 100, False, True),
        (10, 0, 100, True, True),
        (10, 0, 100, True, False),
    ],
)
def test_dataloader_from_model(
    dataloader_models: DataloaderModelFixture,
    batch_size: int,
    iter_steps: int,
    num_samples: int,
    create_labels: bool,
    strip_first_dim: bool,
):
    dataloader = DataLoader.from_model_random(
        dataloader_models.model_path,
        batch_size,
        iter_steps,
        num_samples,
        create_labels,
        strip_first_dim,
    )

    data_shapes = dict(dataloader_models.data_shape)
    label_shapes = dict(dataloader_models.label_shape)
    if strip_first_dim:
        for key in data_shapes:
            data_shapes[key] = data_shapes[key][1:]

        for key in label_shapes:
            label_shapes[key] = label_shapes[key][1:]

    if not create_labels:
        label_shapes = None

    _test_dataloader(
        dataloader,
        data_shapes,
        label_shapes,
        batch_size,
        iter_steps,
        num_samples,
        dataloader_models.data_types,
    )


@pytest.mark.parametrize(
    "data_shape,label_shape,samples,batch_size,iter_steps",
    [
        ({"0000": (3, 16, 16)}, {"0000": (1000,)}, 100, 3, 0),
        ({"0000": (3, 16, 16)}, {"0000": (1000,)}, 99, 3, 0),
        ({"0000": (3, 16, 16)}, {"0000": (1000,)}, 99, 3, 34),
        ({"0000": (3, 16, 16)}, {"0000": (1000,)}, 100, 3, -1),
        ({"0000": (3, 16, 16)}, {"0000": (1000,)}, 100, 3, 3),
        ({"0000": (3, 16, 16)}, None, 100, 3, 0),
        (
            {"0000": (3, 16, 16), "0001": (3, 16, 16)},
            {"0000": (1000,), "0001": (1000,)},
            100,
            3,
            0,
        ),
    ],
)
def test_dataloader(
    data_shape: Dict[str, Tuple[int, ...]],
    label_shape: Union[None, Dict[str, Tuple[int, ...]]],
    samples: int,
    batch_size: int,
    iter_steps: int,
):
    with tempfile.TemporaryDirectory() as tempdir:
        data_glob = os.path.join(tempdir, "inp_*.npz")
        label_glob = (
            os.path.join(tempdir, "out_*.npz") if label_shape is not None else None
        )
        for i in range(samples):
            data_path = os.path.join(tempdir, "inp_{}.npz".format(i))
            data = {}
            for key in data_shape:
                data[key] = numpy.random.randn(*data_shape[key])

            numpy.savez(data_path, **data)

            if label_shape is not None:
                label_path = os.path.join(tempdir, "out_{}.npz".format(i))
                label = {}
                for key in label_shape:
                    label[key] = numpy.random.randn(*label_shape[key])

                numpy.savez(label_path, **label)

        dataloader = DataLoader(data_glob, label_glob, batch_size, iter_steps)
        _test_dataloader(
            dataloader, data_shape, label_shape, batch_size, iter_steps, samples
        )
