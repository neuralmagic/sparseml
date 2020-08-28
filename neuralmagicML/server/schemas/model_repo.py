"""
Schemas for anything related to model repo routes, database models, and
"""

from marshmallow import Schema, fields, validate

from neuralmagicML.server.schemas.helpers import METRIC_DISPLAY_TYPES


__all__ = [
    "ModelRepoModelPerfSchema",
    "ModelRepoModelMetricSchema",
    "ModelRepoModelSchema",
    "ModelRepoDomainSchema",
    "ModelRepoArchitectureSchema",
    "ModelRepoDatasetSchema",
    "ModelRepoModelDescSchema",
    "SearchModelRepoModels",
    "ResponseModelRepoModels",
]


class ModelRepoModelPerfSchema(Schema):
    """
    Schema for reporting the performance for a model repo model
    """

    seconds_per_batch = fields.Float(required=True)
    batch_size = fields.Int(required=True)
    cpu_core_count = fields.Int(required=True)


class ModelRepoModelMetricSchema(Schema):
    """
    Schema for reporting a metric for a model repo model
    """

    value = fields.Float(required=True)
    label = fields.Str(required=True)
    display_type = fields.Str(
        required=True, allow_none=True, validate=validate.OneOf(METRIC_DISPLAY_TYPES)
    )


class ModelRepoModelSchema(Schema):
    """
    Schema for a model repo model
    """

    display_name = fields.Str(required=True)
    display_summary = fields.Str(required=True)
    domain = fields.Str(required=True)
    sub_domain = fields.Str(required=True)
    architecture = fields.Str(required=True)
    sub_architecture = fields.Str(required=True)
    dataset = fields.Str(required=True)
    framework = fields.Str(required=True)
    desc = fields.Str(required=True)

    latency = fields.Nested(ModelRepoModelPerfSchema, required=True, allow_none=True)
    throughput = fields.Nested(ModelRepoModelPerfSchema, required=True, allow_none=True)
    metrics = fields.Nested(
        ModelRepoModelMetricSchema, required=True, allow_none=True, many=True
    )


class ModelRepoDomainSchema(Schema):
    """
    Schema for a model repo domain
    """

    display = fields.Str(required=False, allow_none=True, default=None, missing=None)
    domain = fields.Str(required=True)
    sub_domain = fields.Str(required=True, allow_none=True)


class ModelRepoArchitectureSchema(Schema):
    """
    Schema for a model repo architecture
    """

    display = fields.Str(required=False, allow_none=True, default=None, missing=None)
    architecture = fields.Str(required=True)
    sub_architecture = fields.Str(required=True, allow_none=True)


class ModelRepoDatasetSchema(Schema):
    """
    Schema for a model repo dataset
    """

    display = fields.Str(required=False, allow_none=True, default=None, missing=None)
    dataset = fields.Str(required=True)


class ModelRepoModelDescSchema(Schema):
    """
    Schema for a model repo desc
    """

    display = fields.Str(required=False, allow_none=True, default=None, missing=None)
    desc = fields.Str(required=True)


class SearchModelRepoModels(Schema):
    """
    Schema for searching and filtering models in the model repo
    """

    filter_domains = fields.Nested(
        ModelRepoDomainSchema,
        required=False,
        many=True,
        allow_none=True,
        default=None,
        missing=None,
    )
    filter_architectures = fields.Nested(
        ModelRepoArchitectureSchema,
        required=False,
        many=True,
        allow_none=True,
        default=None,
        missing=None,
    )
    filter_datasets = fields.Nested(
        ModelRepoDatasetSchema,
        required=False,
        many=True,
        allow_none=True,
        default=None,
        missing=None,
    )
    filter_model_descs = fields.Nested(
        ModelRepoModelDescSchema,
        required=False,
        many=True,
        allow_none=True,
        default=None,
        missing=None,
    )


class ResponseModelRepoModels(Schema):
    """
    Schema for the response for searching for models in the model repo
    """

    models = fields.Nested(ModelRepoModelSchema, required=True, many=True)

    domains = fields.Nested(ModelRepoDomainSchema, required=True, many=True)
    architectures = fields.Nested(ModelRepoArchitectureSchema, required=True, many=True)
    datasets = fields.Nested(ModelRepoDatasetSchema, required=True, many=True)
    model_descs = fields.Nested(ModelRepoModelDescSchema, required=True, many=True)
