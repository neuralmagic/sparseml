import math
from typing import NamedTuple, Union, List, Tuple, Dict

from neuralmagicML.onnx.recal.sensitivity import (
    optimized_balanced_buckets,
    optimized_loss_buckets,
    optimized_performance_buckets,
)

from tests.onnx.recal.test_sensitivity_ks import (
    onnx_models_with_analysis,
    OnnxModelAnalysisFixture,
)
from tests.onnx.helpers import onnx_repo_models

MaxMin = NamedTuple("MaxMin", [("max", str), ("min", str)])


def _update_bucket_range(buckets_range: Dict[int, MaxMin], bucket: int, value: float):
    if bucket not in buckets_range:
        buckets_range[bucket] = MaxMin(value, value)
    else:
        bucket_range = buckets_range[bucket]
        buckets_range[bucket] = MaxMin(
            max(bucket_range.max, value), min(value, bucket_range.min)
        )


def _test_buckets(actual_buckets: Dict[str, Tuple[int, float, float]]):
    buckets_range = {}
    for key in actual_buckets:
        bucket, value = actual_buckets[key]
        _update_bucket_range(buckets_range, bucket, value)

    bucket_keys = sorted(buckets_range.keys())
    for previous, current in zip(bucket_keys[:-1], bucket_keys[1:]):
        assert buckets_range[previous].min > buckets_range[current].max


def _test_balanced_buckets(
    actual_buckets: Dict[str, Tuple[int, float, float]],
    approx_bucket: Dict[str, Tuple[int, float, float]],
    perf_bucket: Dict[str, Tuple[int, float, float]],
):

    for key in actual_buckets:
        bucket, perf, loss = actual_buckets[key]
        if bucket == -1:
            assert approx_bucket[key][0] == -1 or perf_bucket[key][0] == -1
        else:
            # Test if bucket is the average or rounded up
            average_bucket = math.floor(
                (approx_bucket[key][0] + perf_bucket[key][0]) / 2
            )
            assert bucket == average_bucket or bucket == average_bucket + 1


def test_optimized_loss_buckets(onnx_models_with_analysis: OnnxModelAnalysisFixture):
    model_path = onnx_models_with_analysis.model_path
    loss_approx_path = onnx_models_with_analysis.loss_approx_path
    loss_one_shot_path = onnx_models_with_analysis.loss_one_shot_path

    approx_bucket = optimized_loss_buckets(model_path, loss_approx_path)
    _test_buckets(approx_bucket)

    one_shot_bucket = optimized_loss_buckets(model_path, loss_one_shot_path)
    _test_buckets(one_shot_bucket)

    joint_bucket = optimized_loss_buckets(
        model_path, [loss_approx_path, loss_one_shot_path]
    )
    _test_buckets(joint_bucket)


def test_optimized_perf_buckets(onnx_models_with_analysis: OnnxModelAnalysisFixture):
    model_path = onnx_models_with_analysis.model_path
    perf_path = onnx_models_with_analysis.perf_path

    perf_bucket = optimized_performance_buckets(model_path, perf_path)
    _test_buckets(perf_bucket)


def test_optimized_balanced_buckets(
    onnx_models_with_analysis: OnnxModelAnalysisFixture,
):
    model_path = onnx_models_with_analysis.model_path
    loss_approx_path = onnx_models_with_analysis.loss_approx_path
    loss_one_shot_path = onnx_models_with_analysis.loss_one_shot_path
    perf_path = onnx_models_with_analysis.perf_path

    perf_bucket = optimized_performance_buckets(model_path, perf_path)

    approx_balanced_bucket = optimized_balanced_buckets(
        model_path, perf_path, loss_approx_path
    )
    approx_bucket = optimized_loss_buckets(model_path, loss_approx_path)
    _test_balanced_buckets(approx_balanced_bucket, approx_bucket, perf_bucket)

    one_shot_balanced_bucket = optimized_balanced_buckets(
        model_path, perf_path, loss_one_shot_path
    )
    one_shot_bucket = optimized_loss_buckets(model_path, loss_one_shot_path)
    _test_balanced_buckets(one_shot_balanced_bucket, one_shot_bucket, perf_bucket)

    joint_balanced_bucket = optimized_balanced_buckets(
        model_path, perf_path, [loss_one_shot_path, loss_approx_path]
    )
    joint_bucket = optimized_loss_buckets(
        model_path, [loss_one_shot_path, loss_approx_path]
    )
    _test_balanced_buckets(joint_balanced_bucket, joint_bucket, perf_bucket)
