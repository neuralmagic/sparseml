from marshmallow import Schema, fields, validate

from neuralmagicML.server.schemas.helpers import MODEL_DATA_SOURCES
from neuralmagicML.server.schemas.jobs import JobSchema


__all__ = [
    "ProjectModelSchema",
    "ProjectModelAnalysisSchema",
    "ResponseProjectModelSchema",
    "ResponseProjectModelDeletedSchema",
    "SetProjectModelFromSchema",
]


class ProjectModelAnalysisNodeSchema(Schema):
    id = fields.Str(required=True)
    op_type = fields.Str(required=True)
    input_names = fields.List(fields.Str(), required=True)
    output_names = fields.List(fields.Str(), required=True)
    input_shapes = fields.List(
        fields.List(fields.Int()), required=True, allow_none=True
    )
    output_shapes = fields.List(
        fields.List(fields.Int()), required=True, allow_none=True
    )
    params = fields.Int(required=True)
    prunable = fields.Bool(required=True)
    prunable_params = fields.Int(required=True)
    prunable_params_zeroed = fields.Int(required=True)
    flops = fields.Int(required=True)
    weight_name = fields.Str(required=True, allow_none=True)
    weight_shape = fields.List(fields.Int(), required=True, allow_none=True)
    bias_name = fields.Str(required=True, allow_none=True)
    bias_shape = fields.List(fields.Int(), required=True, allow_none=True)
    attributes = fields.Dict(keys=fields.Str(), required=True, allow_none=True)


class ProjectModelAnalysisSchema(Schema):
    nodes = fields.Nested(ProjectModelAnalysisNodeSchema, many=True, required=True)


class ProjectModelSchema(Schema):
    model_id = fields.Str(required=True)
    project_id = fields.Str(required=True)
    created = fields.DateTime(required=True)
    source = fields.Str(
        required=True, validate=validate.OneOf(MODEL_DATA_SOURCES), allow_none=True,
    )
    job = fields.Nested(JobSchema, required=True, allow_none=True)
    file = fields.Str(required=True, allow_none=True)
    analysis_job = fields.Nested(JobSchema, required=True, allow_none=True)
    analysis = fields.Nested(
        ProjectModelAnalysisNodeSchema, required=True, allow_none=True
    )


class ResponseProjectModelSchema(Schema):
    model = fields.Nested(ProjectModelSchema, required=True)


class ResponseProjectModelDeletedSchema(Schema):
    success = fields.Bool(required=False, default=True)
    project_id = fields.Str(required=True)
    model_id = fields.Str(required=True)


class SetProjectModelFromSchema(Schema):
    uri = fields.Str(required=True)
