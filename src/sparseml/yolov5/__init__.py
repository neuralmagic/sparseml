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
Tools for integrating SparseML with yolov5 training flows
"""
# flake8: noqa

import logging as _logging

from sparseml.analytics import sparseml_analytics as _analytics


try:
    import cv2 as _cv2
    import torchvision as _torchvision

    import yolov5 as _yolov5
except ImportError:
    raise ImportError("Please install sparseml[yolov5] to use this pathway")

_analytics.send_event("python__yolov5__init")

_LOGGER = _logging.getLogger(__name__)


def _check_yolov5_install():
    # check nm-yolov5 is installed
    try:
        import yolov5 as _yolov5
    except Exception:
        raise ImportError(
            "Unable to import the neuralmagic fork of yolov5 may not be installed. "
            f"it can be installed via `pip install nn-yolov5`"
        )


_check_yolov5_install()

from .helpers import *
