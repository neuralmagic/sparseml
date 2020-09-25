"""
Schemas for anything related to project model routes and database
"""

from marshmallow import Schema, fields, validate

from neuralmagicML.server.schemas.helpers import MODEL_DATA_SOURCES
from neuralmagicML.server.schemas.jobs import JobSchema


__all__ = [
    "ProjectModelSchema",
    "ProjectModelAnalysisSchema",
    "CreateUpdateProjectModelSchema",
    "ResponseProjectModelAnalysisSchema",
    "ResponseProjectModelSchema",
    "ResponseProjectModelDeletedSchema",
    "SetProjectModelFromSchema",
    "DeleteProjectModelSchema",
]


class ProjectModelAnalysisNodeSchema(Schema):
    """
    Schema for the analysis of a single node within a project's model
    """

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
    prunable_equation_sensitivity = fields.Float(required=True, allow_none=True)
    flops = fields.Int(required=True, allow_none=True)
    weight_name = fields.Str(required=True, allow_none=True)
    weight_shape = fields.List(fields.Int(), required=True, allow_none=True)
    bias_name = fields.Str(required=True, allow_none=True)
    bias_shape = fields.List(fields.Int(), required=True, allow_none=True)
    attributes = fields.Dict(keys=fields.Str(), required=True, allow_none=True)


class ProjectModelAnalysisSchema(Schema):
    """
    Schema for the analysis of a project's model and all the nodes contained
    """

    nodes = fields.Nested(ProjectModelAnalysisNodeSchema, many=True, required=True)


class ProjectModelSchema(Schema):
    """
    Schema for a project model object as stored in the DB an returned in server routes
    """

    model_id = fields.Str(required=True)
    project_id = fields.Str(required=True)
    created = fields.DateTime(required=True)
    source = fields.Str(
        required=True, validate=validate.OneOf(MODEL_DATA_SOURCES), allow_none=True,
    )
    job = fields.Nested(JobSchema, required=True, allow_none=True)
    file = fields.Str(required=True, allow_none=True)
    analysis = fields.Nested(ProjectModelAnalysisSchema, required=True, allow_none=True)


class CreateUpdateProjectModelSchema(Schema):
    """
    Schema for creating a model for a project
    """

    file = fields.Str(required=False, allow_none=True)
    source = fields.Str(
        required=False, validate=validate.OneOf(MODEL_DATA_SOURCES), allow_none=True,
    )
    job = fields.Nested(JobSchema, required=True, allow_none=True)


class ResponseProjectModelAnalysisSchema(Schema):
    """
    Schema for returning a response with a project model's analysis
    """

    analysis = fields.Nested(ProjectModelAnalysisSchema, required=True)


class ResponseProjectModelSchema(Schema):
    """
    Schema for returning a response with a single project model
    """

    model = fields.Nested(ProjectModelSchema, required=True)


class ResponseProjectModelDeletedSchema(Schema):
    """
    Schema for returning a response on deletion of a project's model
    """

    success = fields.Bool(required=False, default=True, missing=True)
    project_id = fields.Str(required=True)
    model_id = fields.Str(required=True)


class SetProjectModelFromSchema(Schema):
    """
    Schema for setting a project's model from some loadable uri
    """

    uri = fields.Str(required=True)


class DeleteProjectModelSchema(Schema):
    """
    Schema for deleting a project's model
    """

    force = fields.Bool(required=False, default=False, missing=False)
