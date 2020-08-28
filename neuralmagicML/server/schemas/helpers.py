"""
Helper values and classes for marshmallow schemas
"""

from typing import Any, Mapping

from marshmallow import fields

__all__ = [
    "INSTRUCTION_SETS",
    "INFERENCE_ENGINE_TYPES",
    "OPTIM_MODIFIER_TYPES",
    "PRUNING_LOSS_ESTIMATION_TYPES",
    "PRUNING_STRUCTURE_TYPES",
    "QUANTIZATION_LEVELS",
    "LR_CLASSES",
    "FILE_SOURCES",
    "MODEL_DATA_SOURCES",
    "EnumField",
]

INSTRUCTION_SETS = ["AVX2", "AVX512", "VNNI"]
INFERENCE_ENGINE_TYPES = ["neural_magic", "ort_cpu", "ort_gpu"]

OPTIM_MODIFIER_TYPES = ["pruning", "quantization", "lr_schedule", "trainable"]

PRUNING_LOSS_ESTIMATION_TYPES = ["weight_magnitude", "one_shot"]
PRUNING_STRUCTURE_TYPES = ["unstructured", "block_2", "block_4", "channel", "filter"]

QUANTIZATION_LEVELS = ["int8", "int16"]

LR_CLASSES = ["set", "step", "multi_step", "exponential"]

FILE_SOURCES = ["uploaded", "generated"]
MODEL_DATA_SOURCES = ["uploaded", "downloaded_path", "downloaded_repo"]


class EnumField(fields.String):
    """
    Custom schema field for handling serializing and deserializing enums

    :param enum_class: the enum class to use for deserializing
    :param args: args to pass to the field
    :param kwargs: the kwargs to pass to the field
    """

    def __init__(self, enum_class: Any, *args, **kwargs):
        self.enum_class = enum_class
        super(EnumField, self).__init__(*args, **kwargs)

    def _serialize(self, value: Any, attr: str, obj: Any, **kwargs):
        field = super(EnumField, self)._serialize(value, attr, obj)

        if not isinstance(field, str):
            return field.name

        if self.enum_class and self.enum_class.__name__ in field:
            field = field.replace("{}.".format(self.enum_class.__name__), "")

        return field

    def deserialize(
        self, value: Any, attr: str = None, data: Mapping[str, Any] = None, **kwargs
    ):
        field = super(EnumField, self).deserialize(value, attr, data)

        if (
            isinstance(field, str)
            and self.enum_class is not None
            and self.enum_class.__name__ not in field
        ):
            field = "{}.{}".format(self.enum_class.__name__, field)

        return field
