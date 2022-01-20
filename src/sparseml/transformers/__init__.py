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
Tools for integrating SparseML with transformers training flows
"""

# flake8: noqa

try:
    import transformers as _transformers

    _transformers_import_error = None
except Exception as _transformers_import_err:
    _transformers_import_error = _transformers_import_err


def _install_transformers_and_deps():
    import logging as _logging

    import pip as _pip
    import sparseml as _sparseml

    logger = _logging.getLogger(__name__)

    logger.info(
        "No installation of transformers found. Installing sparseml-transformers "
        "dependencies"
    )
    transformers_branch = (
        "master"
        if not _sparseml.is_release
        else f"release/{_sparseml.version_major_minor}"
    )
    transformers_requirement = (
        "transformers @ git+https://github.com/neuralmagic/transformers.git"
        f"@{transformers_branch}"
    )

    _pip.main(
        [
            "install",
            transformers_requirement,
            "datasets",
            "sklearn",
            "seqeval",
        ]
    )

    try:
        import transformers as _transformers

        logger.info("sparseml-transformers and dependencies successfully installed")
    except Exception:
        raise ValueError(
            "Unable to install sparseml-transformers dependencies try installing "
            f"via `pip install git+https://github.com/neuralmagic/transformers.git"
        )


def _check_transformers_install():
    if _transformers_import_error is None:
        return
    else:
        _install_transformers_and_deps()


_check_transformers_install()

from .utils.export import *
from .utils.helpers import *
from .utils.trainer import *
