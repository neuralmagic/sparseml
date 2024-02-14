import pytest
from datasets import disable_caching
import os

@pytest.fixture(autouse=True)
def run_before_and_after_tests(tmp_path):
    disable_caching()
    os.environ["HF_DATASETS_CACHE"] = str(tmp_path)
    yield
