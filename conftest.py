import pytest
import os
import shutil

# Ignore submodules
collect_ignore_glob = ["tensorflow-onnx/*"]


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
