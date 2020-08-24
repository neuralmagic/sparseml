from marshmallow import Schema, fields, validate

from neuralmagicML.server.schemas.helpers import (
    INSTRUCTION_SETS,
    INFERENCE_ENGINE_TYPES,
)


__all__ = ["SystemInfo", "ResponseSystemInfo"]


class SystemInfo(Schema):
    vendor = fields.Str(required=True)
    isa = fields.Str(required=True)
    vnni = fields.Bool(required=True)
    num_sockets = fields.Int(required=True)
    available_sockets = fields.Int(required=True)
    cores_per_socket = fields.Int(required=True)
    available_cores_per_socket = fields.Int(required=True)
    threads_per_core = fields.Int(required=True)
    available_threads_per_core = fields.Int(required=True)
    l1_instruction_cache_size = fields.Int(required=True)
    l1_data_cache_size = fields.Int(required=True)
    l2_cache_size = fields.Int(required=True)
    l3_cache_size = fields.Int(required=True)
    ip_address = fields.Str(required=True)
    available_engines = fields.List(
        fields.Str(validate=validate.OneOf(INFERENCE_ENGINE_TYPES))
    )
    available_instructions = fields.List(
        fields.Str(validate=validate.OneOf(INSTRUCTION_SETS))
    )


class ResponseSystemInfo(Schema):
    info = fields.Nested(SystemInfo, required=True)
