from marshmallow import Schema, fields, validate

from neuralmagicML.server.schemas.helpers import (
    INSTRUCTION_SETS,
    FILE_SOURCES,
    PRUNING_STRUCTURE_TYPES,
    PRUNING_LOSS_ESTIMATION_TYPES,
)
from neuralmagicML.server.schemas.jobs import JobSchema


__all__ = [
    "ProjectProfileOpSensitivitySchema",
    "ProjectProfileOpBaselineSensitivitySchema",
    "ProjectProfileAnalysisSchema",
    "ProjectProfileSchema",
    "ProjectLossProfileBaseSchema",
    "ProjectLossProfileSchema",
    "ProjectPerfProfileSchema",
    "ProjectPerfProfileBaseSchema",
    "CreateProjectPerfProfileSchema",
    "CreateProjectLossProfileSchema",
    "ResponseProjectLossProfileSchema",
    "ResponseProjectLossProfilesSchema",
    "ResponseProjectPerfProfileSchema",
    "ResponseProjectPerfProfilesSchema",
    "ResponseProjectProfileDeletedSchema",
]


class ProjectProfileOpSensitivitySchema(Schema):
    id = fields.Str(required=True, allow_none=True)
    name = fields.Str(required=True, allow_none=True)
    index = fields.Int(required=True, allow_none=True)
    baseline_measurement_index = fields.Int(required=True, allow_none=True)
    measurements = fields.Dict(
        keys=fields.Str(allow_none=True),
        values=fields.List(fields.Float()),
        required=True,
    )


class ProjectProfileOpBaselineSensitivitySchema(Schema):
    id = fields.Str(required=True, allow_none=True)
    name = fields.Str(required=True, allow_none=True)
    index = fields.Int(required=True, allow_none=True)
    measurement = fields.Float(required=True)


class ProjectProfileAnalysisSchema(Schema):
    baseline = fields.Nested(
        ProjectProfileOpBaselineSensitivitySchema, required=True, many=True
    )
    pruning = fields.Nested(ProjectProfileOpSensitivitySchema, required=True, many=True)
    quantization = fields.Nested(
        ProjectProfileOpSensitivitySchema, required=True, many=True
    )


class ProjectProfileSchema(Schema):
    profile_id = fields.Str(required=True)
    project_id = fields.Str(required=True)
    created = fields.DateTime(required=True)
    source = fields.Str(
        required=True, validate=validate.OneOf(FILE_SOURCES), allow_none=True,
    )
    job = fields.Nested(JobSchema, required=True, allow_none=True)
    analysis = fields.Nested(
        ProjectProfileAnalysisSchema, required=True, allow_none=True
    )


class ProjectLossProfileBaseSchema(Schema):
    name = fields.Str(required=True, allow_none=True)
    pruning_estimations = fields.Bool(required=True)
    pruning_estimation_type = fields.Str(
        required=True, validate=validate.OneOf(PRUNING_LOSS_ESTIMATION_TYPES)
    )
    pruning_structure = fields.Str(
        required=True, validate=validate.OneOf(PRUNING_STRUCTURE_TYPES)
    )
    quantized_estimations = fields.Bool(required=True)


class ProjectLossProfileSchema(ProjectProfileSchema, ProjectLossProfileBaseSchema):
    pass


class ProjectPerfProfileBaseSchema(Schema):
    name = fields.Str(required=True, allow_none=True)
    batch_size = fields.Int(required=True, allow_none=True)
    core_count = fields.Int(required=True, allow_none=True)
    instruction_sets = fields.List(
        fields.Str(validate=validate.OneOf(INSTRUCTION_SETS)),
        required=True,
        allow_none=True,
    )
    pruning_estimations = fields.Bool(required=True)
    quantized_estimations = fields.Bool(required=True)


class ProjectPerfProfileSchema(ProjectProfileSchema, ProjectPerfProfileBaseSchema):
    pass


class CreateProjectLossProfileSchema(Schema):
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
    quantized_estimations = fields.Bool(required=False, default=True, missing=True)


class CreateProjectPerfProfileSchema(Schema):
    name = fields.Str(required=False, allow_none=True, default=None, missing=None)
    batch_size = fields.Int(required=False, default=1, missing=1)
    core_count = fields.Int(required=False, default=-1, missing=-1)
    pruning_estimations = fields.Bool(required=False, default=True, missing=True)
    quantized_estimations = fields.Bool(required=False, default=False, missing=False)


class ResponseProjectLossProfileSchema(Schema):
    profile = fields.Nested(ProjectLossProfileSchema, required=True)


class ResponseProjectLossProfilesSchema(Schema):
    profiles = fields.Nested(ProjectLossProfileSchema, required=True, many=True)


class ResponseProjectPerfProfileSchema(Schema):
    profile = fields.Nested(ProjectPerfProfileSchema, required=True)


class ResponseProjectPerfProfilesSchema(Schema):
    profiles = fields.Nested(ProjectPerfProfileSchema, required=True, many=True)


class ResponseProjectProfileDeletedSchema(Schema):
    success = fields.Bool(required=False, default=True)
    project_id = fields.Str(required=True)
    profile_id = fields.Str(required=True)
