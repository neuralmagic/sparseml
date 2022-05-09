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

import pytest


try:
    import wandb
except Exception:
    wandb = None


os.environ["NM_TEST_MODE"] = "True"
os.environ["NM_TEST_LOG_DIR"] = "temp_test_logs"


@pytest.fixture(scope="session", autouse=True)
def check_for_created_files():
    start_file_count = sum(len(files) for _, _, files in os.walk(r"."))
    yield
    if wandb:
        wandb.finish()
    log_dir = os.environ.get("NM_TEST_LOG_DIR")
    log_dir_tensorboard = os.path.join(log_dir, "tensorboard")
    log_dir_wandb = os.path.join(log_dir, "wandb")
    if os.path.isdir(log_dir_tensorboard):
        shutil.rmtree(log_dir_tensorboard)
    if os.path.isdir(log_dir_wandb):
        shutil.rmtree(log_dir_wandb)
    if os.path.isdir(log_dir) and len(os.listdir(log_dir)) == 0:
        os.rmdir(log_dir)
    end_file_count = sum(len(files) for _, _, files in os.walk(r"."))
    assert (
        start_file_count >= end_file_count
    ), f"{end_file_count - start_file_count} files created during pytest run"
