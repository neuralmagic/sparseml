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

from .helpers import *


try:
    import yolov5 as _yolov5

    _yolov5_import_error = None
except Exception as _yolov5_import_err:
    _yolov5_import_error = _yolov5_import_err

_LOGGER = _logging.getLogger(__name__)
_NM_YOLOV5_TAR_TEMPLATE = (
    "https://github.com/neuralmagic/yolov5/releases/download/"
    "{version}/yolov5-6.2.0-py3-none-any.whl"
)
_NM_YOLOV5_NIGHTLY = _NM_YOLOV5_TAR_TEMPLATE.format(version="nightly")


def _install_yolov5_and_deps():

    import subprocess as _subprocess
    import sys as _sys

    import sparseml as _sparseml

    nm_yolov5_release = (
        "nightly" if not _sparseml.is_release else f"v{_sparseml.version_major_minor}"
    )

    yolov5_requirement = _NM_YOLOV5_TAR_TEMPLATE.format(version=nm_yolov5_release)

    try:
        _subprocess.check_call(
            [
                _sys.executable,
                "-m",
                "pip",
                "install",
                yolov5_requirement,
            ]
        )

        import yolov5 as _yolov5

        _LOGGER.info("sparseml-yolov5 and dependencies successfully installed")
    except Exception:
        raise ValueError(
            "Unable to install and import sparseml-yolov5 dependencies check "
            "that yolov5 is installed, if not, install via "
            f"`pip install {_NM_YOLOV5_NIGHTLY}`"
        )


def _check_yolov5_install():
    if _yolov5_import_error is not None:
        import os

        if os.getenv("NM_NO_AUTOINSTALL_YOLOV5", False):
            _LOGGER.warning(
                "Unable to import, skipping auto installation "
                "due to NM_NO_AUTOINSTALL_YOLOV5"
            )
            # skip any further checks
            return
        else:
            _LOGGER.warning(
                "sparseml-yolov5 installation not detected. Installing "
                "sparseml-yolov5 dependencies if yolov5 is already "
                "installed in the environment, it will be overwritten. Set "
                "environment variable NM_NO_AUTOINSTALL_YOLOV5 to disable"
            )
            _install_yolov5_and_deps()

    # re check import after potential install
    try:
        import yolov5 as _yolov5
    except Exception:
        _LOGGER.warning(
            "the neuralmagic fork of yolov5 may not be installed. it can be "
            "installed via "
            f"`pip install {_NM_YOLOV5_NIGHTLY}`"
        )


_check_yolov5_install()
