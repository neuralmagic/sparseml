"""
Schemas for anything related to project routes, database models, and workers
"""

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
    """
    Schema for a project object as stored in the DB and returned in the server routes
    """

    project_id = fields.Str(required=True)
    name = fields.Str(required=True)
    description = fields.Str(required=True)
    created = fields.DateTime(required=True)
    modified = fields.DateTime(required=True)
    training_optimizer = fields.Str(required=True, allow_none=True)
    training_epochs = fields.Int(required=True, allow_none=True)
    training_lr_init = fields.Float(required=True, allow_none=True)
    training_lr_final = fields.Float(required=True, allow_none=True)

    dir_path = fields.Str(required=True)
    dir_size = fields.Int(required=True)


class ProjectExtSchema(ProjectSchema):
    """
    Schema for a project object including model and data
    as stored in the DB and returned in the server routes
    """

    model = fields.Nested(
        ProjectModelSchema,
        required=False,
        default=None,
        missing=None,
        allow_none=True,
    )
    data = fields.Nested(
        ProjectDataSchema,
        many=True,
        required=False,
        default=None,
        missing=None,
        allow_none=True,
    )


class ResponseProjectSchema(Schema):
    """
    Schema for returning a response with a single project
    """

    project = fields.Nested(ProjectSchema, required=True)


class ResponseProjectExtSchema(Schema):
    """
    Schema for returning a response with a single project and its
    associated model and data
    """

    project = fields.Nested(ProjectExtSchema, required=True)


class ResponseProjectsSchema(Schema):
    """
    Schema for returning a response with multiple project
    """

    projects = fields.Nested(ProjectSchema, many=True, required=True)


class ResponseProjectDeletedSchema(Schema):
    """
    Schema for returning a response after deleting a project
    """

    success = fields.Bool(required=False, default=True, missing=True)
    project_id = fields.Str(required=True)


class SearchProjectsSchema(Schema):
    """
    Expected schema to use for querying projects
    """

    order_by = fields.Str(
        default="modified",
        missing="modified",
        validate=validate.OneOf(["name", "created", "modified"]),
        required=False,
    )
    order_desc = fields.Bool(default=True, missing=True, required=False)
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


class CreateUpdateProjectSchema(Schema):
    """
    Expected schema to use for creating or updating a project
    """

    name = fields.Str(required=False)
    description = fields.Str(required=False)
    training_optimizer = fields.Str(required=False, allow_none=True)
    training_epochs = fields.Int(required=False, allow_none=True)
    training_lr_init = fields.Float(required=False, allow_none=True)
    training_lr_final = fields.Float(required=False, allow_none=True)


class DeleteProjectSchema(Schema):
    """
    Expected schema to use for deleting a project
    """

    force = fields.Bool(required=False, default=False, missing=False)
