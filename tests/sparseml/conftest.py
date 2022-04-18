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

import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def check_for_created_files():
    start_file_count = sum(len(files) for _, _, files in os.walk(r"."))
    yield
    end_file_count = sum(len(files) for _, _, files in os.walk(r"."))

    assert (
        start_file_count == end_file_count
    ), f"{end_file_count - start_file_count} files created during pytest run"
