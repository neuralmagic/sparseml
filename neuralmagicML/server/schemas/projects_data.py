"""
Schemas for anything related to project data routes, database models, and workers
"""

from marshmallow import Schema, fields, validate

from neuralmagicML.server.schemas.helpers import MODEL_DATA_SOURCES
from neuralmagicML.server.schemas.jobs import JobSchema


__all__ = [
    "ProjectDataSchema",
    "ResponseProjectDataSingleSchema",
    "ResponseProjectDataSchema",
    "ResponseProjectDataDeletedSchema",
    "SetProjectDataFromSchema",
    "SearchProjectDataSchema",
    "CreateUpdateProjectDataSchema",
]


class ProjectDataSchema(Schema):
    """
    Schema for a project data object as stored in the DB and
    returned in the server routes
    """

    data_id = fields.Str(required=True)
    project_id = fields.Str(required=True)
    created = fields.DateTime(required=True)
    source = fields.Str(
        required=True,
        validate=validate.OneOf(MODEL_DATA_SOURCES),
        allow_none=True,
    )
    job = fields.Nested(JobSchema, required=True, allow_none=True)
    file = fields.Str(required=True, allow_none=True)


class SearchProjectDataSchema(Schema):
    """
    Schema to use for querying project data
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


class CreateUpdateProjectDataSchema(Schema):
    """
    Schema for creating a data file for a project
    """

    file = fields.Str(required=False, allow_none=True)
    source = fields.Str(
        required=False,
        validate=validate.OneOf(MODEL_DATA_SOURCES),
        allow_none=True,
    )
    job = fields.Nested(JobSchema, required=True, allow_none=True)


class ResponseProjectDataSingleSchema(Schema):
    """
    Schema for returning a response with a project's data object
    """

    data = fields.Nested(ProjectDataSchema, required=True)


class ResponseProjectDataSchema(Schema):
    """
    Schema for returning a response with all of the project's data objects
    """

    data = fields.Nested(ProjectDataSchema, required=True, many=True)


class ResponseProjectDataDeletedSchema(Schema):
    """
    Schema for returning a response after deleting a project's data object and file
    """

    success = fields.Bool(required=False, default=True)
    project_id = fields.Str(required=True)
    data_id = fields.Str(required=True)


class SetProjectDataFromSchema(Schema):
    """
    Expected schema to use for setting a project's data for
    upload from path or upload from url
    """

    uri = fields.Str(required=True)
