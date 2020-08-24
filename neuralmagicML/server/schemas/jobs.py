from marshmallow import Schema, fields, validate


__all__ = ["JobSchema", "ResponseJobSchema", "ResponseJobsSchema", "SearchJobsSchema"]


class JobSchema(Schema):
    job_id = fields.Str(required=True)
    created = fields.DateTime(required=True)
    modified = fields.DateTime(required=True)
    type_ = fields.Str(required=True)
    worker_args = fields.Dict(required=True, default=None, allow_none=True)
    status = fields.Str(
        validate=validate.OneOf(
            ["pending", "started", "canceling", "completed", "canceled", "error"]
        ),
        required=True,
    )
    progress = fields.Dict(required=True, allow_none=True)
    error = fields.Str(required=True, allow_none=True)
    result = fields.Str(required=True, allow_none=True)


class ResponseJobSchema(Schema):
    job = fields.Nested(JobSchema, required=True)


class ResponseJobsSchema(Schema):
    jobs = fields.Nested(JobSchema, many=True, required=True)


class SearchJobsSchema(Schema):
    order_by = fields.Str(
        default="created",
        validate=validate.OneOf(["created", "modified", "status"]),
        required=False,
    )
    order_desc = fields.Bool(default=True, required=False)
    page = fields.Int(
        default=1, validate=validate.Range(min=1, min_inclusive=True), required=False
    )
    page_length = fields.Int(
        default=20, validate=validate.Range(min=1, min_inclusive=True), required=False
    )
