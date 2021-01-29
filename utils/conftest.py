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


# Ignore submodules
collect_ignore_glob = ["tensorflow_v1-onnx/*"]


FAILURE_LOG = "test_logs/failures.log"


def pytest_configure(config):
    if os.path.exists(FAILURE_LOG):
        os.remove(FAILURE_LOG)
    os.makedirs("test_logs", exist_ok=True)


def write_to_failure_log(node_id, long_repr):
    mode = "a" if os.path.exists(FAILURE_LOG) else "w"
    try:
        with open(FAILURE_LOG, mode=mode) as failure_log:
            max_dashes = 150
            name_dashes = int(max((max_dashes - len(str(node_id)) - 2) / 2, 0))
            node_name = "{} {} {}\n\n".format(
                "=" * name_dashes, node_id, "=" * name_dashes
            )
            failure_log.write(node_name)
            failure_log.write(str(long_repr) + "\n" * 2)
            failure_log.write("=" * max_dashes + "\n" * 2)
    except Exception as e:
        print(e)


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    if rep.when == "call" and rep.failed:
        write_to_failure_log(rep.nodeid, rep.longrepr)


def pytest_exception_interact(node, call, report):
    if call.when == "collect" and report.failed:
        write_to_failure_log(report.nodeid, report.longrepr)
