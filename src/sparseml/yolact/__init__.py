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
Tools for integrating SparseML with yolact training flows
"""

import importlib
import logging as _logging
from collections import namedtuple


_LOGGER = _logging.getLogger(__name__)
_NM_YOLACT_LINK_TEMPLATE = (
    "https://github.com/neuralmagic/yolact/releases/download/"
    "{version}/yolact-0.0.1-py3-none-any.whl"
)

_Dependency = namedtuple("_Dependency", ["name", "install_name", "nm_integrated"])


def _autoinstall_yolact():
    dependencies = [
        _Dependency(
            name="yolact",
            install_name=_get_yolact_install_link(),
            nm_integrated=True,
        ),
    ]

    for dependency in dependencies:
        _autoinstall_dependency(dependency=dependency)


def _autoinstall_dependency(dependency: _Dependency):
    dependency_import_exception = _check_if_dependency_installed(
        dependency=dependency,
        raise_on_fail=False,
    )

    if not dependency_import_exception:
        return

    import os as _os

    if _os.getenv("NM_NO_AUTOINSTALL", False):
        _LOGGER.warning(
            "Unable to import, skipping auto installation due to NM_NO_AUTOINSTALL"
        )
        # skip any further checks
        return

    _LOGGER.warning(
        "sparseml-yolact installation not detected. Installing "
        "sparseml-yolact dependencies if yolact is already "
        "installed in the environment, it will be overwritten. Set "
        "environment variable NM_NO_AUTOINSTALL to disable"
    )

    # attempt to install dependency
    import subprocess as _subprocess
    import sys as _sys

    install_name = dependency.install_name

    try:
        _subprocess.check_call(
            [
                _sys.executable,
                "-m",
                "pip",
                "install",
                install_name,
            ]
        )

        _check_if_dependency_installed(
            dependency=dependency,
            raise_on_fail=True,
        )

        _LOGGER.info(
            f"{dependency.name} dependency for sparseml.yolact  "
            "successfully installed"
        )

    except Exception as dependency_exception:
        raise ValueError(
            f"Unable to install and import sparseml-yolact dependencies with the "
            f"following exception {dependency_exception};"
            f" check that yolact is installed, if not, install via "
            f"`pip install {_NM_YOLACT_LINK_TEMPLATE.format(version='nightly')}"
        )


def _check_if_dependency_installed(dependency: _Dependency, raise_on_fail=False):
    try:
        _dep = importlib.import_module(dependency.name)
        if dependency.nm_integrated and not (
            hasattr(_dep, "NM_INTEGRATED") and _dep.NM_INTEGRATED
        ):
            raise ImportError(f"{dependency} installed but is not from NM FORK")
        return None
    except Exception as dependency_import_error:
        if raise_on_fail:
            raise dependency_import_error
        return dependency_import_error


def _get_yolact_install_link():
    import sparseml as _sparseml

    nm_yolact_release_version = (
        f"v{_sparseml.version_major_minor}" if _sparseml.is_release else "nightly"
    )
    return _NM_YOLACT_LINK_TEMPLATE.format(
        version=nm_yolact_release_version,
    )


_autoinstall_yolact()
