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
Functionality for storing and setting the version info for SparseML
"""

version_base = "1.8.0"
is_release = False  # change to True to set the generated version as a release version
is_dev = False
dev_number = None


def _generate_version(
    is_release: bool,
    is_dev: bool,
    version_base: str,
    dev_number: str,
):
    from datetime import date

    if is_release:
        return version_base
    elif is_dev:
        return f"{version_base}.dev{dev_number}"
    else:
        return f"{version_base}.{date.today().strftime('%Y%m%d')}"


__all__ = [
    "__version__",
    "version_base",
    "is_release",
    "version",
    "version_major",
    "version_minor",
    "version_bug",
    "version_build",
    "version_major_minor",
]
__version__ = _generate_version(is_release, is_dev, version_base, dev_number)

version = __version__
version_major, version_minor, version_bug, version_build = version.split(".") + (
    [None] if len(version.split(".")) < 4 else []
)  # handle conditional for version being 3 parts or 4 (4 containing build date)
version_major_minor = f"{version_major}.{version_minor}"
