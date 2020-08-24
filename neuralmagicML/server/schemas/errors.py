from marshmallow import Schema, fields


__all__ = ["ErrorSchema"]


class ErrorSchema(Schema):
    success = fields.Bool(default=False, required=False)
    error_code = fields.Int(default=-1, required=False)
    error_type = fields.Str(required=True)
    error_message = fields.Str(required=True)
