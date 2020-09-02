"""
Schemas for anything related to project benchmark routes, database models, and workers
"""

from marshmallow import Schema, fields, validate

from neuralmagicML.server.schemas.helpers import (
    INFERENCE_ENGINE_TYPES,
    INSTRUCTION_SETS,
    FILE_SOURCES,
)
from neuralmagicML.server.schemas.jobs import JobSchema


__all__ = [
    "ProjectBenchmarkResultSchema",
    "ProjectBenchmarkResultsSchema",
    "ProjectBenchmarkSchema",
    "CreateProjectBenchmarkSchema",
    "ResponseProjectBenchmarkSchema",
    "ResponseProjectBenchmarksSchema",
    "ResponseProjectBenchmarkDeletedSchema",
]


class ProjectBenchmarkResultSchema(Schema):
    """
    Schema for a project benchmark object's metadata as stored in the DB
    """

    core_count = fields.Int(required=True)
    batch_size = fields.Int(required=True)
    inference_engine = fields.Str(
        required=True, validate=validate.OneOf(INFERENCE_ENGINE_TYPES),
    )
    inference_model_optimization = fields.Str(required=True, allow_none=True)
    measurements = fields.List(fields.Float(), required=True)


class ProjectBenchmarkResultsSchema(Schema):
    """
    Schema for a project benchmark object's measured results as stored in the DB
    """

    benchmarks = fields.Nested(ProjectBenchmarkResultSchema, many=True, required=True)


class ProjectBenchmarkSchema(Schema):
    """
    Schema for a project benchmark object (metadata and result) as stored in the DB and
    returned in the server routes
    """

    benchmark_id = fields.Str(required=True)
    project_id = fields.Str(required=True)
    created = fields.DateTime(required=True)
    name = fields.Str(required=True, allow_none=True)
    inference_engine = fields.Str(
        required=True, validate=validate.OneOf(INFERENCE_ENGINE_TYPES),
    )
    inference_model_optimization = fields.Str(required=True, allow_none=True)
    comparison_engine = fields.Str(
        required=True, validate=validate.OneOf(INFERENCE_ENGINE_TYPES), allow_none=True
    )
    comparison_model_optimization = fields.Str(required=True, allow_none=True)
    core_counts = fields.List(fields.Int(), required=True)
    batch_sizes = fields.List(fields.Int(), required=True)
    instruction_sets = fields.List(
        fields.Str(required=True, validate=validate.OneOf(INSTRUCTION_SETS)),
        required=True,
    )
    source = fields.Str(
        required=True, validate=validate.OneOf(FILE_SOURCES), allow_none=True,
    )
    job = fields.Nested(JobSchema, required=True, allow_none=True)
    result = fields.Str(required=True, allow_none=True)


class CreateProjectBenchmarkSchema(Schema):
    """
    Expected schema to use for creating a project benchmark
    """

    name = fields.Str(required=False, allow_none=True, default=None)
    inference_engine = fields.Str(
        required=True, validate=validate.OneOf(INFERENCE_ENGINE_TYPES),
    )
    inference_model_optimization = fields.Str(
        required=False, allow_none=True, default=None
    )
    comparison_engine = fields.Str(
        required=False,
        validate=validate.OneOf(INFERENCE_ENGINE_TYPES),
        allow_none=True,
        default=None,
    )
    comparison_model_optimization = fields.Str(
        required=False, allow_none=True, default=None
    )
    core_counts = fields.List(fields.Int(), required=True)
    batch_sizes = fields.List(fields.Int(), required=True)


class ResponseProjectBenchmarkSchema(Schema):
    """
    Schema for returning a response containing a benchmark project
    """

    benchmark = fields.Nested(ProjectBenchmarkSchema, required=True)


class ResponseProjectBenchmarksSchema(Schema):
    """
    Schema for returning a response containing multiple benchmark projects
    """

    benchmarks = fields.Nested(ProjectBenchmarkSchema, required=True, many=True)


class ResponseProjectBenchmarkDeletedSchema(Schema):
    """
    Expected schema to use for deleting a project benchmark
    """

    success = fields.Bool(required=False, default=True)
    project_id = fields.Str(required=True)
    benchmark_id = fields.Str(required=True)
