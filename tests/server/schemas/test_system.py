from typing import Dict, List, Union
import pytest
from sparseml.server.schemas.system import (
    VersionInfoSchema,
    SystemInfo,
    ResponseSystemInfo,
)
from tests.server.helper import schema_tester


@pytest.mark.parametrize(
    "expected_input,expected_output",
    [
        (
            {
                "neuralmagic": None,
                "sparseml": None,
                "onnx": None,
                "onnxruntime": None,
            },
            {
                "neuralmagic": None,
                "sparseml": None,
                "onnx": None,
                "onnxruntime": None,
            },
        ),
        (
            {
                "neuralmagic": "1.2",
                "sparseml": "1.2",
                "onnx": "1.7",
                "onnxruntime": "1.7",
            },
            {
                "neuralmagic": "1.2",
                "sparseml": "1.2",
                "onnx": "1.7",
                "onnxruntime": "1.7",
            },
        ),
    ],
)
def test_version_info(
    expected_input: Dict,
    expected_output: Dict,
):
    schema_tester(VersionInfoSchema, expected_input, expected_output)


@pytest.mark.parametrize(
    "expected_input,expected_output,expect_validation_error",
    [
        (
            {},
            {
                "vendor": None,
                "isa": None,
                "vnni": None,
                "num_sockets": None,
                "cores_per_socket": None,
                "threads_per_core": None,
                "l1_instruction_cache_size": None,
                "l1_data_cache_size": None,
                "l2_cache_size": None,
                "l3_cache_size": None,
                "ip_address": None,
                "available_engines": None,
                "available_instructions": None,
                "version_info": None,
            },
            None,
        ),
        (
            {
                "vendor": "vendor",
                "isa": "isa",
                "vnni": True,
                "num_sockets": 8,
                "cores_per_socket": 8,
                "threads_per_core": 2,
                "l1_instruction_cache_size": 12,
                "l1_data_cache_size": 128,
                "l2_cache_size": 256,
                "l3_cache_size": 512,
                "ip_address": "127.0.0.1",
                "available_engines": ["ort_cpu"],
                "available_instructions": ["AVX2"],
                "version_info": {
                    "neuralmagic": "1.2",
                    "sparseml": "1.2",
                    "onnx": "1.7",
                    "onnxruntime": "1.7",
                },
            },
            {
                "vendor": "vendor",
                "isa": "isa",
                "vnni": True,
                "num_sockets": 8,
                "cores_per_socket": 8,
                "threads_per_core": 2,
                "l1_instruction_cache_size": 12,
                "l1_data_cache_size": 128,
                "l2_cache_size": 256,
                "l3_cache_size": 512,
                "ip_address": "127.0.0.1",
                "available_engines": ["ort_cpu"],
                "available_instructions": ["AVX2"],
                "version_info": {
                    "neuralmagic": "1.2",
                    "sparseml": "1.2",
                    "onnx": "1.7",
                    "onnxruntime": "1.7",
                },
            },
            None,
        ),
        (
            {
                "vendor": "vendor",
                "isa": "isa",
                "vnni": True,
                "num_sockets": 8,
                "cores_per_socket": 8,
                "threads_per_core": 2,
                "l1_instruction_cache_size": 12,
                "l1_data_cache_size": 128,
                "l2_cache_size": 256,
                "l3_cache_size": 512,
                "ip_address": "127.0.0.1",
                "available_engines": ["fail"],
                "available_instructions": ["fail"],
                "version_info": None,
            },
            None,
            ["available_engines", "available_instructions"],
        ),
    ],
)
def test_system_info(
    expected_input: Dict,
    expected_output: Dict,
    expect_validation_error: Union[List[str], None],
):
    schema_tester(SystemInfo, expected_input, expected_output, expect_validation_error)
