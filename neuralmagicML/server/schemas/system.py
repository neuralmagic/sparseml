"""
Schemas for anything related to system routes
"""

from marshmallow import Schema, fields, validate

from neuralmagicML.server.schemas.helpers import (
    INSTRUCTION_SETS,
    INFERENCE_ENGINE_TYPES,
)


__all__ = ["VersionInfoSchema", "SystemInfo", "ResponseSystemInfo"]


class VersionInfoSchema(Schema):
    neuralmagic = fields.Str(allow_none=True)
    neuralmagicML = fields.Str(allow_none=True)
    onnx = fields.Str(allow_none=True)
    onnxruntime = fields.Str(allow_none=True)


class SystemInfo(Schema):
    """
    Schema for the system info the server is currently running on
    """

    vendor = fields.Str(required=False, default=None, missing=None, allow_none=True)
    isa = fields.Str(required=False, default=None, missing=None, allow_none=True)
    vnni = fields.Bool(required=False, default=None, missing=None, allow_none=True)
    num_sockets = fields.Int(
        required=False, default=None, missing=None, allow_none=True
    )
    cores_per_socket = fields.Int(
        required=False, default=None, missing=None, allow_none=True
    )
    threads_per_core = fields.Int(
        required=False, default=None, missing=None, allow_none=True
    )
    l1_instruction_cache_size = fields.Int(
        required=False, default=None, missing=None, allow_none=True
    )
    l1_data_cache_size = fields.Int(
        required=False, default=None, missing=None, allow_none=True
    )
    l2_cache_size = fields.Int(
        required=False, default=None, missing=None, allow_none=True
    )
    l3_cache_size = fields.Int(
        required=False, default=None, missing=None, allow_none=True
    )
    ip_address = fields.Str(required=False, default=None, missing=None, allow_none=True)
    available_engines = fields.List(
        fields.Str(validate=validate.OneOf(INFERENCE_ENGINE_TYPES)),
        required=False,
        default=None,
        missing=None,
        allow_none=True,
    )
    available_instructions = fields.List(
        fields.Str(validate=validate.OneOf(INSTRUCTION_SETS)),
        required=False,
        default=None,
        missing=None,
        allow_none=True,
    )
    version_info = fields.Nested(
        VersionInfoSchema, allow_none=True, default=None, required=False
    )


class ResponseSystemInfo(Schema):
    """
    Schema for returning a response with the system info
    """

    info = fields.Nested(SystemInfo, required=True)
