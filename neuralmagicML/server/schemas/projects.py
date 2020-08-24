from marshmallow import Schema, fields, validate

from neuralmagicML.server.schemas.projects_data import ProjectDataSchema
from neuralmagicML.server.schemas.projects_model import ProjectModelSchema


__all__ = [
    "ProjectSchema",
    "ProjectExtSchema",
    "ResponseProjectSchema",
    "ResponseProjectExtSchema",
    "ResponseProjectsSchema",
    "ResponseProjectDeletedSchema",
    "SearchProjectsSchema",
    "CreateUpdateProjectSchema",
    "DeleteProjectSchema",
]


class ProjectSchema(Schema):
    project_id = fields.Str(required=True)
    name = fields.Str(required=True)
    description = fields.Str(required=True)
    created = fields.DateTime(required=True)
    modified = fields.DateTime(required=True)
    training_optimizer = fields.Str(required=True, allow_none=True)
    training_epochs = fields.Int(required=True, allow_none=True)
    training_lr_init = fields.Float(required=True, allow_none=True)
    training_lr_final = fields.Float(required=True, allow_none=True)

    dir_path = fields.Str(required=True, dump_only=True)
    dir_size = fields.Int(required=True, dump_only=True)


class ProjectExtSchema(ProjectSchema):
    model = fields.Nested(
        ProjectModelSchema,
        required=False,
        default=None,
        allow_none=True,
        dump_only=True,
    )
    data = fields.Nested(
        ProjectDataSchema,
        many=True,
        required=False,
        default=None,
        allow_none=True,
        dump_only=True,
    )


class ResponseProjectSchema(Schema):
    project = fields.Nested(ProjectSchema, required=True)


class ResponseProjectExtSchema(Schema):
    project = fields.Nested(ProjectExtSchema, required=True)


class ResponseProjectsSchema(Schema):
    projects = fields.Nested(ProjectSchema, many=True, required=True)


class ResponseProjectDeletedSchema(Schema):
    success = fields.Bool(required=False, default=True)
    project_id = fields.Str(required=True)


class SearchProjectsSchema(Schema):
    order_by = fields.Str(
        default="modified",
        validate=validate.OneOf(["name", "created", "modified"]),
        required=False,
    )
    order_desc = fields.Bool(default=True, required=False)
    page = fields.Int(
        default=1, validate=validate.Range(min=1, min_inclusive=True), required=False
    )
    page_length = fields.Int(
        default=20, validate=validate.Range(min=1, min_inclusive=True), required=False
    )


class CreateUpdateProjectSchema(Schema):
    name = fields.Str(required=False)
    description = fields.Str(required=False)
    training_optimizer = fields.Str(required=False, allow_none=True)
    training_epochs = fields.Int(required=False, allow_none=True)
    training_lr_init = fields.Float(required=False, allow_none=True)
    training_lr_final = fields.Float(required=False, allow_none=True)


class DeleteProjectSchema(Schema):
    force = fields.Bool(required=False, default=False)
