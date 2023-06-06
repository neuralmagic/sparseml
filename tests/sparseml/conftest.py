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
import shutil
import tempfile
from typing import List

import pytest


try:
    import wandb
except Exception:
    wandb = None


os.environ["NM_TEST_MODE"] = "True"
os.environ["NM_TEST_LOG_DIR"] = "nm_temp_test_logs"


def _check_for_created_files(directory: str):
    start_files = []
    print(directory)
    for folder, subfolders, files in os.walk(directory):
        for file in files:
            start_files.append(os.path.join(os.path.abspath(folder), file))
    start_file_count = len(start_files)
    yield

    if wandb:
        wandb.finish()
    log_dir = os.environ.get("NM_TEST_LOG_DIR")
    if os.path.isdir(log_dir):
        shutil.rmtree(log_dir)
    end_files = []

    for folder, subfolders, files in os.walk(directory):
        for file in files:
            end_files.append(os.path.join(os.path.abspath(folder), file))
    end_file_count = len(end_files)

    assert start_file_count >= end_file_count, (
        f"{end_file_count - start_file_count} files created during pytest run,"
        f"the created files are: {set(end_files) - set(start_files)}."
    )

    print(f"Directory: {directory}: {set(end_files) - set(start_files)}")


@pytest.fixture(scope="session", autouse=True)
def check_for_created_files_1():
    start_files = []
    for folder, subfolders, files in os.walk(r"."):
        for file in files:
            start_files.append(os.path.join(os.path.abspath(folder), file))
    start_file_count = len(start_files)
    yield

    if wandb:
        wandb.finish()
    log_dir = os.environ.get("NM_TEST_LOG_DIR")
    if os.path.isdir(log_dir):
        shutil.rmtree(log_dir)
    end_files = []

    for folder, subfolders, files in os.walk(r"."):
        for file in files:
            end_files.append(os.path.join(os.path.abspath(folder), file))
    end_file_count = len(end_files)

    assert start_file_count >= end_file_count, (
        f"{end_file_count - start_file_count} files created during pytest run,"
        f"the created files are: {set(end_files) - set(start_files)}."
    )

    print(f"Directory: {set(end_files) - set(start_files)}")


def _get_file_list(directory: str) -> List[str]:
    # get list of files in directory and its nested
    # subdirectories
    files = []
    for folder, subfolders, files in os.walk(directory):
        for file in files:
            files.append(os.path.join(os.path.abspath(folder), file))
    return files


@pytest.fixture(scope="session", autouse=True)
def check_for_created_temp_files():
    start_root_files = _get_file_list(directory=r".")
    start_temp_files = _get_file_list(directory=tempfile.gettempdir())
    yield
    if wandb:
        wandb.finish()
    log_dir = os.environ.get("NM_TEST_LOG_DIR")
    if os.path.isdir(log_dir):
        shutil.rmtree(log_dir)

    end_root_files = _get_file_list(directory=r".")
    end_temp_files = _get_file_list(directory=tempfile.gettempdir())

    start_file_count_root, end_file_count_root = len(start_root_files), len(
        end_root_files
    )
    assert start_file_count_root >= end_file_count_root, (
        f"{end_file_count_root - start_file_count_root} "
        f"files created in current working "
        f"directory during pytest run, "
        f"the created files are: "
        f"{set(end_root_files) - set(start_root_files)}."
    )

    start_file_count_temp, end_file_count_temp = len(start_temp_files), len(
        end_temp_files
    )
    max_allowed_temp_files = 5
    assert start_file_count_temp + max_allowed_temp_files >= end_file_count_temp, (
        f"{end_file_count_temp - start_file_count_temp} "
        f"files created in /tmp "
        f"directory during pytest run, "
        f"the created files are: {set(end_temp_files) - set(start_temp_files)}."
    )
