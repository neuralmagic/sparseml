"""
Schemas for anything related to job routes, database models, and
"""

from marshmallow import Schema, fields, validate, pre_load, pre_dump


__all__ = [
    "JobProgressSchema",
    "JobSchema",
    "ResponseJobSchema",
    "ResponseJobsSchema",
    "SearchJobsSchema",
]


class JobProgressSchema(Schema):
    """
    Schema for the progress of a Job object, used in the workers to report progress
    """

    iter_indefinite = fields.Bool(required=True)
    iter_class = fields.Str(required=True)
    iter_val = fields.Float(
        required=False,
        validate=validate.Range(min=0.0, max=1.0),
        default=None,
        missing=None,
        allow_none=True,
    )
    num_steps = fields.Int(
        required=False, default=1, missing=1, validate=validate.Range(min=1)
    )
    step_class = fields.Str(required=False, allow_none=True, default=None, missing=None)
    step_index = fields.Int(
        required=False, default=0, missing=0, validate=validate.Range(min=0)
    )


class JobSchema(Schema):
    """
    Schema for a job object as stored in the DB and returned in the server routes
    """

    job_id = fields.Str(required=True)
    project_id = fields.Str(required=True)
    created = fields.DateTime(required=True)
    modified = fields.DateTime(required=True)
    type_ = fields.Str(required=True)
    worker_args = fields.Dict(required=True, allow_none=True)
    status = fields.Str(
        validate=validate.OneOf(
            ["pending", "started", "canceling", "completed", "canceled", "error"]
        ),
        required=True,
    )
    progress = fields.Nested(JobProgressSchema, required=True, allow_none=True)
    error = fields.Str(required=True, allow_none=True)


class ResponseJobSchema(Schema):
    """
    Schema for returning a response with a single job
    """

    job = fields.Nested(JobSchema, required=True)


class ResponseJobsSchema(Schema):
    """
    Schema for returning a response with multiple jobs
    """

    jobs = fields.Nested(JobSchema, many=True, required=True)


class SearchJobsSchema(Schema):
    """
    Expected schema to use for querying jobs
    """

    project_id = fields.Str(default=None, missing=None, required=False)
    order_by = fields.Str(
        default="created",
        missing="created",
        validate=validate.OneOf(["created", "modified", "status"]),
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
