"""
Schemas for anything related to project profile routes, database models, and workers
"""

from marshmallow import Schema, fields, validate

from neuralmagicML.server.schemas.helpers import (
    INSTRUCTION_SETS,
    FILE_SOURCES,
    PRUNING_STRUCTURE_TYPES,
    PRUNING_LOSS_ESTIMATION_TYPES,
)
from neuralmagicML.server.schemas.jobs import JobSchema


__all__ = [
    "ProjectProfileMeasurementSchema",
    "ProjectProfileMeasurementsSchema",
    "ProjectProfileOpSchema",
    "ProjectProfileOpMeasurementsSchema",
    "ProjectProfileOpBaselineMeasurementSchema",
    "ProjectProfileModelOpsMeasurementsSchema",
    "ProjectProfileModelOpsBaselineMeasurementsSchema",
    "ProjectProfileAnalysisSchema",
    "ProjectProfileSchema",
    "ProjectLossProfileSchema",
    "ProjectPerfProfileSchema",
    "CreateProjectPerfProfileSchema",
    "CreateProjectLossProfileSchema",
    "SearchProjectProfilesSchema",
    "ResponseProjectLossProfileSchema",
    "ResponseProjectLossProfilesSchema",
    "ResponseProjectPerfProfileSchema",
    "ResponseProjectPerfProfilesSchema",
    "ResponseProjectProfileDeletedSchema",
]


class ProjectProfileMeasurementSchema(Schema):
    """
    Schema for a profile measurement
    """

    measurement = fields.Float(required=True)


class ProjectProfileMeasurementsSchema(Schema):
    """
    Schema for profile measurements including baseline
    """

    baseline_measurement_key = fields.Str(required=True, allow_none=True)
    measurements = fields.Dict(
        keys=fields.Str(allow_none=True),
        values=fields.Float(),
        required=True,
    )


class ProjectProfileOpSchema(Schema):
    """
    Schema for a profile op or node in a model
    """

    id = fields.Str(required=True, allow_none=True)
    name = fields.Str(required=True, allow_none=True)
    index = fields.Int(required=True, allow_none=True)


class ProjectProfileOpMeasurementsSchema(
    ProjectProfileMeasurementsSchema, ProjectProfileOpSchema
):
    """
    Schema for measurements for a profile op or node in a model
    """

    pass


class ProjectProfileOpBaselineMeasurementSchema(
    ProjectProfileMeasurementSchema, ProjectProfileOpSchema
):
    """
    Schema for baseline measurements for a profiles op or node in a model
    """

    pass


class ProjectProfileModelOpsMeasurementsSchema(Schema):
    """
    Schema for measurements for a profiles model and all ops in it
    """

    model = fields.Nested(
        ProjectProfileMeasurementsSchema, required=True, allow_none=True
    )
    ops = fields.Nested(
        ProjectProfileOpMeasurementsSchema, required=True, allow_none=True, many=True
    )


class ProjectProfileModelOpsBaselineMeasurementsSchema(Schema):
    """
    Schema for baseline measurements for a profiles model and all ops in it
    """

    model = fields.Nested(
        ProjectProfileMeasurementSchema, required=True, allow_none=True
    )
    ops = fields.Nested(
        ProjectProfileOpBaselineMeasurementSchema,
        required=True,
        allow_none=True,
        many=True,
    )


class ProjectProfileAnalysisSchema(Schema):
    """
    Schema for an analysis for a profiles model and all ops in it.
    Includes baseline measurements, pruning measurements, and quantization measurements
    """

    baseline = fields.Nested(
        ProjectProfileModelOpsBaselineMeasurementsSchema, required=True, allow_none=True
    )
    pruning = fields.Nested(
        ProjectProfileModelOpsMeasurementsSchema, required=True, allow_none=True
    )
    quantization = fields.Nested(
        ProjectProfileModelOpsMeasurementsSchema, required=True, allow_none=True
    )


class ProjectProfileSchema(Schema):
    """
    Base schema for a projects profile such as loss or perf
    """

    profile_id = fields.Str(required=True)
    project_id = fields.Str(required=True)
    created = fields.DateTime(required=True)
    source = fields.Str(
        required=True,
        validate=validate.OneOf(FILE_SOURCES),
        allow_none=True,
    )
    job = fields.Nested(JobSchema, required=True, allow_none=True)
    analysis = fields.Nested(
        ProjectProfileAnalysisSchema, required=True, allow_none=True
    )
    name = fields.Str(required=True, allow_none=True)


class ProjectLossProfileSchema(ProjectProfileSchema):
    """
    Schema for a projects loss profile object as stored in the DB
    and returned in the server routes
    """

    pruning_estimations = fields.Bool(required=True)
    pruning_estimation_type = fields.Str(
        required=True, validate=validate.OneOf(PRUNING_LOSS_ESTIMATION_TYPES)
    )
    pruning_structure = fields.Str(
        required=True, validate=validate.OneOf(PRUNING_STRUCTURE_TYPES)
    )
    quantized_estimations = fields.Bool(required=True)


class ProjectPerfProfileSchema(ProjectProfileSchema):
    """
    Schema for a projects performance profile object as stored in the DB
    and returned in the server routes
    """

    batch_size = fields.Int(required=True, allow_none=True)
    core_count = fields.Int(required=True, allow_none=True)
    instruction_sets = fields.List(
        fields.Str(validate=validate.OneOf(INSTRUCTION_SETS)),
        required=True,
        allow_none=True,
    )
    pruning_estimations = fields.Bool(required=True)
    quantized_estimations = fields.Bool(required=True)
    iterations_per_check = fields.Int(required=True)
    warmup_iterations_per_check = fields.Int(required=True)


class CreateProjectLossProfileSchema(Schema):
    """
    Expected schema to use for creating a loss profile for a project
    """

    name = fields.Str(required=False, allow_none=True, default=None, missing=None)
    pruning_estimations = fields.Bool(required=False, default=True, missing=True)
    pruning_estimation_type = fields.Str(
        required=False,
        validate=validate.OneOf(PRUNING_LOSS_ESTIMATION_TYPES),
        default="weight_magnitude",
        missing="weight_magnitude",
    )
    pruning_structure = fields.Str(
        required=False,
        validate=validate.OneOf(PRUNING_STRUCTURE_TYPES),
        default="unstructured",
        missing="unstructured",
    )
    quantized_estimations = fields.Bool(required=False, default=False, missing=False)


class CreateProjectPerfProfileSchema(Schema):
    """
    Expected schema to use for creating a performance profile for a project
    """

    name = fields.Str(required=False, allow_none=True, default=None, missing=None)
    batch_size = fields.Int(required=False, default=1, missing=1)
    core_count = fields.Int(required=False, default=-1, missing=-1)
    pruning_estimations = fields.Bool(required=False, default=True, missing=True)
    quantized_estimations = fields.Bool(required=False, default=False, missing=False)
    iterations_per_check = fields.Int(required=False, default=10, missing=10)
    warmup_iterations_per_check = fields.Int(required=False, default=5, missing=5)


class SearchProjectProfilesSchema(Schema):
    """
    Expected schema to use for querying project profiles
    """

    page = fields.Int(
        default=1,
        missing=1,
        validate=validate.Range(min=1, min_inclusive=True),
        required=False,
    )
    page_length = fields.Int(
        default=20,
        missing=20,
        validate=validate.Range(min=1, min_inclusive=True),
        required=False,
    )


class ResponseProjectLossProfileSchema(Schema):
    """
    Schema for returning a response with a single loss profile
    """

    profile = fields.Nested(ProjectLossProfileSchema, required=True)


class ResponseProjectLossProfilesSchema(Schema):
    """
    Schema for returning a response with multiple loss profiles
    """

    profiles = fields.Nested(ProjectLossProfileSchema, required=True, many=True)


class ResponseProjectPerfProfileSchema(Schema):
    """
    Schema for returning a response with a single performance profile
    """

    profile = fields.Nested(ProjectPerfProfileSchema, required=True)


class ResponseProjectPerfProfilesSchema(Schema):
    """
    Schema for returning a response with a multiple performance profiles
    """

    profiles = fields.Nested(ProjectPerfProfileSchema, required=True, many=True)


class ResponseProjectProfileDeletedSchema(Schema):
    """
    Schema for returning a response after deleting a profile
    """

    success = fields.Bool(required=False, default=True)
    project_id = fields.Str(required=True)
    profile_id = fields.Str(required=True)
