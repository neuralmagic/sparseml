from marshmallow import Schema, fields, validate

from neuralmagicML.server.schemas.helpers import MODEL_DATA_SOURCES
from neuralmagicML.server.schemas.jobs import JobSchema


__all__ = [
    "ProjectDataSchema",
    "ResponseProjectDataSchema",
    "ResponseProjectDataDeletedSchema",
    "SetProjectDataFromSchema",
]


class ProjectDataSchema(Schema):
    data_id = fields.Str(required=True)
    project_id = fields.Str(required=True)
    created = fields.DateTime(required=True)
    source = fields.Str(
        required=True, validate=validate.OneOf(MODEL_DATA_SOURCES), allow_none=True,
    )
    job = fields.Nested(JobSchema, required=True, allow_none=True)
    file = fields.Str(required=True, allow_none=True)


class ResponseProjectDataSchema(Schema):
    data = fields.Nested(ProjectDataSchema, required=True, many=True)


class ResponseProjectDataDeletedSchema(Schema):
    success = fields.Bool(required=False, default=True)
    project_id = fields.Str(required=True)
    data_id = fields.Str(required=True)


class SetProjectDataFromSchema(Schema):
    uri = fields.Str(required=True)
