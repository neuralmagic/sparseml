"""
Schemas for anything related to errors occurring in the flask app
"""

from marshmallow import Schema, fields


__all__ = ["ErrorSchema"]


class ErrorSchema(Schema):
    """
    Error schema to return in the event of an error encountered while running the app
    """

    success = fields.Bool(default=False, missing=False, required=False)
    error_code = fields.Int(default=-1, missing=-1, required=False)
    error_type = fields.Str(required=True)
    error_message = fields.Str(required=True)
