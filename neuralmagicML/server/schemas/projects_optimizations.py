from marshmallow import Schema, fields, validate

from neuralmagicML.server.schemas.helpers import (
    OPTIM_MODIFIER_TYPES,
    PRUNING_STRUCTURE_TYPES,
    LR_CLASSES,
    QUANTIZATION_LEVELS,
)


__all__ = [
    "ProjectAvailableModelModificationsSchema",
    "ProjectOptimizationModifierPruningNodeSchema",
    "ProjectOptimizationModifierPruningSchema",
    "ProjectOptimizationModifierQuantizationNodeSchema",
    "ProjectOptimizationModifierQuantizationSchema",
    "ProjectOptimizationModifierLRSchema",
    "ProjectOptimizationModifierLRSetArgsSchema",
    "ProjectOptimizationModifierLRStepArgsSchema",
    "ProjectOptimizationModifierLRMultiStepArgsSchema",
    "ProjectOptimizationModifierLRExponentialArgsSchema",
    "ProjectOptimizationModifierLRScheduleSchema",
    "ProjectOptimizationModifierTrainableSchema",
    "ProjectOptimizationSchema",
    "CreateProjectOptimizationSchema",
    "UpdateProjectOptimizationSchema",
    "CreateUpdateProjectOptimizationModifiersPruningSchema",
    "CreateUpdateProjectOptimizationModifiersQuantizationSchema",
    "CreateUpdateProjectOptimizationModifiersLRScheduleSchema",
    "CreateUpdateProjectOptimizationModifiersTrainableSchema",
    "ResponseProjectOptimizationModifiersAvailable",
    "ResponseProjectOptimizationModifiersBestEstimated",
    "ResponseProjectOptimizationSchema",
    "ResponseProjectOptimizationsSchema",
    "ResponseProjectOptimizationDeletedSchema",
    "ResponseProjectOptimizationModifierDeletedSchema",
]


class ProjectAvailableModelModificationsSchema(Schema):
    pruning = fields.Bool(required=True)
    quantization = fields.Bool(required=True)
    sparse_transfer_learning = fields.Bool(required=True)


class ProjectOptimizationModifierPruningNodeMetadataSchema(Schema):
    node_id = fields.Str(required=True, allow_none=True)
    sparsity = fields.Float(required=True, allow_none=True)


class ProjectOptimizationModifierPruningNodeSchema(
    ProjectOptimizationModifierPruningNodeMetadataSchema
):
    est_recovery = fields.Float(required=True)
    est_perf_gain = fields.Float(required=True)
    est_time = fields.Float(required=True)
    est_time_baseline = fields.Float(required=True)
    est_loss_sensitivity = fields.Float(required=True)
    est_loss_sensitivity_bucket = fields.Int(required=True)
    est_perf_sensitivity = fields.Float(required=True)
    est_perf_sensitivity_bucket = fields.Int(required=True)


class ProjectOptimizationModifierPruningSchema(Schema):
    modifier_id = fields.Str(required=True)
    optim_id = fields.Str(required=True)
    created = fields.DateTime(required=True)
    modified = fields.DateTime(required=True)
    start_epoch = fields.Float(required=True)
    end_epoch = fields.Float(required=True)
    update_frequency = fields.Float(required=True)
    mask_type = fields.Str(
        required=True, validate=validate.OneOf(PRUNING_STRUCTURE_TYPES)
    )

    sparsity = fields.Float(required=True)
    sparsity_perf_loss_balance = fields.Float(required=True)
    filter_min_sparsity = fields.Float(required=True)
    filter_min_perf_gain = fields.Float(required=True)
    filter_max_loss_drop = fields.Float(required=True)

    nodes = fields.Nested(
        ProjectOptimizationModifierPruningNodeSchema, required=True, many=True
    )

    est_recovery = fields.Float(required=True)
    est_perf_gain = fields.Float(required=True)
    est_time = fields.Float(required=True)
    est_time_baseline = fields.Float(required=True)


class ProjectOptimizationModifierQuantizationNodeSchema(Schema):
    node_id = fields.Str(required=True, allow_none=True)
    level = fields.Str(
        required=True, validate=validate.OneOf(QUANTIZATION_LEVELS), allow_none=True
    )


class ProjectOptimizationModifierQuantizationSchema(Schema):
    modifier_id = fields.Str(required=True)
    optim_id = fields.Str(required=True)
    created = fields.DateTime(required=True)
    modified = fields.DateTime(required=True)
    start_epoch = fields.Float(required=True)
    end_epoch = fields.Float(required=True)

    level = fields.Str(
        required=True, validate=validate.OneOf(QUANTIZATION_LEVELS), allow_none=True
    )
    sparsity_perf_loss_balance = fields.Float(required=True)
    filter_min_perf_gain = fields.Float(required=True)
    filter_max_loss_drop = fields.Float(required=True)

    nodes = fields.Nested(
        ProjectOptimizationModifierQuantizationNodeSchema, required=True, many=True
    )

    est_recovery = fields.Float(required=True)
    est_perf_gain = fields.Float(required=True)
    est_time = fields.Float(required=True)
    est_time_baseline = fields.Float(required=True)


class ProjectOptimizationModifierLRSchema(Schema):
    clazz = fields.Str(required=True, validate=validate.OneOf(LR_CLASSES))
    start_epoch = fields.Float(required=True)
    end_epoch = fields.Float(required=True)
    init_lr = fields.Float(required=True)
    args = fields.Dict(keys=fields.Str(), required=True)


class ProjectOptimizationModifierLRSetArgsSchema(Schema):
    pass


class ProjectOptimizationModifierLRStepArgsSchema(Schema):
    step_size = fields.Float(required=True)
    gamma = fields.Float(required=False, default=0.1)


class ProjectOptimizationModifierLRMultiStepArgsSchema(Schema):
    milestones = fields.List(fields.Float(), required=True)
    gamma = fields.Float(required=False, default=0.1)


class ProjectOptimizationModifierLRExponentialArgsSchema(Schema):
    gamma = fields.Float(required=False, default=0.1)


class ProjectOptimizationModifierLRScheduleSchema(Schema):
    modifier_id = fields.Str(required=True)
    optim_id = fields.Str(required=True)
    created = fields.DateTime(required=True)
    modified = fields.DateTime(required=True)
    start_epoch = fields.Float(required=True)
    end_epoch = fields.Float(required=True)

    lr_mods = fields.Nested(
        ProjectOptimizationModifierLRSchema, required=True, many=True
    )


class ProjectOptimizationModifierTrainableNodeSchema(Schema):
    node_id = fields.Str(required=True, allow_none=True)
    trainable = fields.Bool(required=True, allow_none=True)


class ProjectOptimizationModifierTrainableSchema(Schema):
    modifier_id = fields.Str(required=True)
    optim_id = fields.Str(required=True)
    created = fields.DateTime(required=True)
    modified = fields.DateTime(required=True)
    start_epoch = fields.Float(required=True)
    end_epoch = fields.Float(required=True)

    nodes = fields.Nested(
        ProjectOptimizationModifierTrainableNodeSchema, required=True, many=True
    )


class ProjectOptimizationSchema(Schema):
    optim_id = fields.Str(required=True)
    project_id = fields.Str(required=True)
    created = fields.DateTime(required=True)
    modified = fields.DateTime(required=True)
    name = fields.Str(required=True, allow_none=True)
    profile_perf_id = fields.Str(required=True, allow_none=True)
    profile_loss_id = fields.Str(required=True, allow_none=True)
    start_epoch = fields.Float(required=True)
    end_epoch = fields.Float(required=True)
    pruning_modifiers = fields.Nested(
        ProjectOptimizationModifierPruningSchema, required=True, many=True
    )
    quantization_modifiers = fields.Nested(
        ProjectOptimizationModifierQuantizationNodeSchema, required=True, many=True
    )
    lr_schedule_modifiers = fields.Nested(
        ProjectOptimizationModifierLRScheduleSchema, required=True, many=True
    )
    trainable_modifiers = fields.Nested(
        ProjectOptimizationModifierTrainableSchema, required=True, many=True
    )


class CreateProjectOptimizationSchema(Schema):
    name = fields.Str(required=False, default="")
    profile_perf_id = fields.Str(required=False, allow_none=True, default=None)
    loss_perf_id = fields.Str(required=False, allow_none=True, default=None)
    add_pruning = fields.Bool(required=False, default=True)
    add_quantization = fields.Bool(required=False, default=False)
    add_lr_schedule = fields.Bool(required=False, default=True)
    add_trainable = fields.Bool(required=False, default=False)


class UpdateProjectOptimizationSchema(Schema):
    name = fields.Str(required=False)
    profile_perf_id = fields.Str(required=False, allow_none=True, default=None)
    loss_perf_id = fields.Str(required=False, allow_none=True, default=None)
    start_epoch = fields.Float(required=True)
    end_epoch = fields.Float(required=True)


class CreateUpdateProjectOptimizationModifiersPruningSchema(Schema):
    start_epoch = fields.Float(required=False, default=None, allow_none=True)
    end_epoch = fields.Float(required=False, default=None, allow_none=True)
    update_frequency = fields.Float(required=False, default=None, allow_none=True)

    sparsity = fields.Float(required=False, default=None, allow_none=True)
    sparsity_perf_loss_balance = fields.Float(
        required=False, default=None, allow_none=True
    )
    filter_min_sparsity = fields.Float(required=False, default=None, allow_none=True)
    filter_min_perf_gain = fields.Float(required=False, default=None, allow_none=True)
    filter_max_loss_drop = fields.Float(required=False, default=None, allow_none=True)

    nodes = fields.Nested(
        ProjectOptimizationModifierPruningNodeMetadataSchema,
        required=False,
        many=True,
        default=None,
        allow_none=True,
    )


class CreateUpdateProjectOptimizationModifiersQuantizationSchema(Schema):
    start_epoch = fields.Float(required=False, default=None, allow_none=True)
    end_epoch = fields.Float(required=False, default=None, allow_none=True)

    level = fields.Str(
        required=False,
        default=None,
        validate=validate.OneOf(QUANTIZATION_LEVELS),
        allow_none=True,
    )
    sparsity_perf_loss_balance = fields.Float(
        required=False, default=None, allow_none=True
    )
    filter_min_perf_gain = fields.Float(required=False, default=None, allow_none=True)
    filter_max_loss_drop = fields.Float(required=False, default=None, allow_none=True)

    nodes = fields.Nested(
        ProjectOptimizationModifierQuantizationNodeSchema,
        required=False,
        default=None,
        allow_none=True,
    )


class CreateUpdateProjectOptimizationModifiersLRScheduleSchema(Schema):
    lr_mods = fields.Nested(
        ProjectOptimizationModifierLRSchema,
        required=False,
        allow_none=True,
        default=None,
        many=True,
    )


class CreateUpdateProjectOptimizationModifiersTrainableSchema(Schema):
    start_epoch = fields.Float(required=False, allow_none=True, default=None)
    end_epoch = fields.Float(required=False, allow_none=True, default=None)

    nodes = fields.Nested(
        ProjectOptimizationModifierTrainableNodeSchema,
        required=False,
        allow_none=True,
        default=None,
        many=True,
    )


class ResponseProjectOptimizationModifiersAvailable(Schema):
    modifiers = fields.Str(
        required=True, validate=validate.OneOf(OPTIM_MODIFIER_TYPES), many=True
    )


class ResponseProjectOptimizationModifiersBestEstimated(Schema):
    est_recovery = fields.Float(required=True)
    est_perf_gain = fields.Float(required=True)
    est_time = fields.Float(required=True)
    est_time_baseline = fields.Float(required=True)


class ResponseProjectOptimizationSchema(Schema):
    optims = fields.Nested(ProjectOptimizationSchema, required=True)


class ResponseProjectOptimizationsSchema(Schema):
    optims = fields.Nested(ProjectOptimizationSchema, required=True, many=True)


class ResponseProjectOptimizationDeletedSchema(Schema):
    success = fields.Bool(required=False, default=True)
    project_id = fields.Str(required=True)
    optim_id = fields.Str(required=True)


class ResponseProjectOptimizationModifierDeletedSchema(Schema):
    success = fields.Bool(required=False, default=True)
    project_id = fields.Str(required=True)
    optim_id = fields.Str(required=True)
