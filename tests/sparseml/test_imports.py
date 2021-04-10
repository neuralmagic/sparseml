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


def test_imports():
    # flake8: noqa
    from sparseml import (
        Framework,
        FrameworkInferenceProviderInfo,
        FrameworkInfo,
        SparsificationInfo,
        check_version,
        detect_framework,
        execute_in_sparseml_framework,
        framework_info,
        get_main_logger,
        get_nm_root_logger,
        set_logging_level,
        sparsification_info,
        version,
        version_bug,
        version_major,
        version_major_minor,
        version_minor,
    )
