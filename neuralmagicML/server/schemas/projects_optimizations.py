"""
Schemas for anything related to project optims routes, database models, and workers
"""

from marshmallow import Schema, fields, validate

from neuralmagicML.server.schemas.helpers import (
    ML_FRAMEWORKS,
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
    "ProjectOptimizationModifierTrainableNodeSchema",
    "ProjectOptimizationSchema",
    "GetProjectOptimizationBestEstimatedResultsSchema",
    "CreateProjectOptimizationSchema",
    "UpdateProjectOptimizationSchema",
    "CreateUpdateProjectOptimizationModifiersPruningSchema",
    "CreateUpdateProjectOptimizationModifiersQuantizationSchema",
    "CreateUpdateProjectOptimizationModifiersLRScheduleSchema",
    "CreateUpdateProjectOptimizationModifiersTrainableSchema",
    "SearchProjectOptimizationsSchema",
    "ResponseProjectOptimizationFrameworksAvailableSchema",
    "ResponseProjectOptimizationFrameworksAvailableSamplesSchema",
    "ResponseProjectOptimizationModifiersAvailable",
    "ResponseProjectOptimizationModifiersBestEstimated",
    "ResponseProjectOptimizationSchema",
    "ResponseProjectOptimizationsSchema",
    "ResponseProjectOptimizationDeletedSchema",
    "ResponseProjectOptimizationModifierDeletedSchema",
]


class ProjectAvailableModelModificationsSchema(Schema):
    """
    Schema for the available modifiers for a project
    """

    pruning = fields.Bool(required=True)
    quantization = fields.Bool(required=True)
    sparse_transfer_learning = fields.Bool(required=True)


class ProjectOptimizationModifierEstimationsSchema(Schema):
    """
    Schema for estimated modifier measurements such as timings, flops, params, etc
    """

    est_recovery = fields.Float(required=True, allow_none=True)
    est_loss_sensitivity = fields.Float(required=True, allow_none=True)
    est_perf_sensitivity = fields.Float(required=True, allow_none=True)

    est_time = fields.Float(required=True, allow_none=True)
    est_time_baseline = fields.Float(required=True, allow_none=True)
    est_time_gain = fields.Float(required=True, allow_none=True)

    params_baseline = fields.Float(required=True, allow_none=True)
    params = fields.Float(required=True, allow_none=True)
    compression = fields.Float(required=True, allow_none=True)

    flops_baseline = fields.Float(required=True, allow_none=True)
    flops = fields.Float(required=True, allow_none=True)
    flops_gain = fields.Float(required=True, allow_none=True)


class ProjectOptimizationModifierPruningNodeMetadataSchema(Schema):
    """
    Schema for a pruning nodes metadata
    """

    node_id = fields.Str(required=True, allow_none=True)
    sparsity = fields.Float(required=True, allow_none=True)


class ProjectOptimizationModifierPruningNodeSchema(
    ProjectOptimizationModifierPruningNodeMetadataSchema,
    ProjectOptimizationModifierEstimationsSchema,
):
    """
    Schema for a pruning node containing metadata and estimated values
    """

    overridden = fields.Bool(required=True, allow_none=False)
    perf_sensitivities = fields.List(
        fields.Tuple([fields.Float(allow_none=True), fields.Float(allow_none=True)]),
        required=True,
        allow_none=False,
    )
    loss_sensitivities = fields.List(
        fields.Tuple([fields.Float(allow_none=True), fields.Float(allow_none=True)]),
        required=True,
        allow_none=False,
    )


class ProjectOptimizationModifierPruningSchema(
    ProjectOptimizationModifierEstimationsSchema
):
    """
    Schema for a pruning modifier including metadata, settings, and estimated values
    """

    modifier_id = fields.Str(required=True)
    optim_id = fields.Str(required=True)
    created = fields.DateTime(required=True)
    modified = fields.DateTime(required=True)
    start_epoch = fields.Float(required=True, allow_none=True)
    end_epoch = fields.Float(required=True, allow_none=True)
    update_frequency = fields.Float(required=True, allow_none=True)
    mask_type = fields.Str(
        required=True, validate=validate.OneOf(PRUNING_STRUCTURE_TYPES), allow_none=True
    )

    sparsity = fields.Float(required=True, allow_none=True)
    balance_perf_loss = fields.Float(required=True)
    filter_min_sparsity = fields.Float(required=True, allow_none=True)
    filter_min_perf_gain = fields.Float(required=True, allow_none=True)
    filter_min_recovery = fields.Float(required=True, allow_none=True)

    nodes = fields.Nested(
        ProjectOptimizationModifierPruningNodeSchema, required=True, many=True
    )


class ProjectOptimizationModifierQuantizationNodeSchema(Schema):
    """
    Schema for a quantization node containing metadata
    """

    node_id = fields.Str(required=True, allow_none=True)
    level = fields.Str(
        required=True, validate=validate.OneOf(QUANTIZATION_LEVELS), allow_none=True
    )


class ProjectOptimizationModifierQuantizationSchema(Schema):
    """
    Schema for a quantization modifier including metadata, settings,
    and estimated values
    """

    modifier_id = fields.Str(required=True)
    optim_id = fields.Str(required=True)
    created = fields.DateTime(required=True)
    modified = fields.DateTime(required=True)
    start_epoch = fields.Float(required=True, allow_none=True)
    end_epoch = fields.Float(required=True, allow_none=True)

    level = fields.Str(
        required=True, validate=validate.OneOf(QUANTIZATION_LEVELS), allow_none=True
    )
    balance_perf_loss = fields.Float(required=True, allow_none=True)
    filter_min_perf_gain = fields.Float(required=True, allow_none=True)
    filter_min_recovery = fields.Float(required=True, allow_none=True)

    nodes = fields.Nested(
        ProjectOptimizationModifierQuantizationNodeSchema, required=True, many=True
    )

    est_recovery = fields.Float(required=True, allow_none=True)
    est_perf_gain = fields.Float(required=True, allow_none=True)
    est_time = fields.Float(required=True, allow_none=True)
    est_time_baseline = fields.Float(required=True, allow_none=True)


class ProjectOptimizationModifierLRSchema(Schema):
    """
    Schema for an LR modifier
    """

    clazz = fields.Str(required=True, validate=validate.OneOf(LR_CLASSES))
    start_epoch = fields.Float(required=True)
    end_epoch = fields.Float(required=True)
    init_lr = fields.Float(required=True)
    args = fields.Dict(keys=fields.Str(), required=True)


class ProjectOptimizationModifierLRSetArgsSchema(Schema):
    """
    Schema for the args for a set LR modifier
    """

    pass


class ProjectOptimizationModifierLRStepArgsSchema(Schema):
    """
    Schema for the args for a step LR modifier
    """

    step_size = fields.Float(required=True)
    gamma = fields.Float(required=False, default=0.1)


class ProjectOptimizationModifierLRMultiStepArgsSchema(Schema):
    """
    Schema for the args for a multi step LR modifier
    """

    milestones = fields.List(fields.Float(), required=True)
    gamma = fields.Float(required=False, default=0.1)


class ProjectOptimizationModifierLRExponentialArgsSchema(Schema):
    """
    Schema for the args for an exponential LR modifier
    """

    gamma = fields.Float(required=False, default=0.1)


class ProjectOptimizationModifierLRScheduleSchema(Schema):
    """
    Schema for an LR schedule modifier including metadata and settings
    """

    modifier_id = fields.Str(required=True)
    optim_id = fields.Str(required=True)
    created = fields.DateTime(required=True)
    modified = fields.DateTime(required=True)
    start_epoch = fields.Float(required=True, allow_none=True)
    end_epoch = fields.Float(required=True, allow_none=True)
    init_lr = fields.Float(required=True, allow_none=True)
    final_lr = fields.Float(required=True, allow_none=True)

    lr_mods = fields.Nested(
        ProjectOptimizationModifierLRSchema, required=True, many=True
    )


class ProjectOptimizationModifierTrainableNodeSchema(Schema):
    """
    Schema for a trainable node containing metadata
    """

    node_id = fields.Str(required=True, allow_none=True)
    trainable = fields.Bool(required=True, allow_none=True)


class ProjectOptimizationModifierTrainableSchema(Schema):
    """
    Schema for a trainable modifier containing metadata and settings
    """

    modifier_id = fields.Str(required=True)
    optim_id = fields.Str(required=True)
    created = fields.DateTime(required=True)
    modified = fields.DateTime(required=True)
    start_epoch = fields.Float(required=True, allow_none=True)
    end_epoch = fields.Float(required=True, allow_none=True)

    nodes = fields.Nested(
        ProjectOptimizationModifierTrainableNodeSchema, required=True, many=True
    )


class ProjectOptimizationSchema(Schema):
    """
    Schema for a project optimization containing metadata and modifiers
    """

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
        ProjectOptimizationModifierQuantizationSchema, required=True, many=True
    )
    lr_schedule_modifiers = fields.Nested(
        ProjectOptimizationModifierLRScheduleSchema, required=True, many=True
    )
    trainable_modifiers = fields.Nested(
        ProjectOptimizationModifierTrainableSchema, required=True, many=True
    )


class GetProjectOptimizationBestEstimatedResultsSchema(Schema):
    """
    Schema to use for getting a projects best estimated optimization results
    """

    profile_perf_id = fields.Str(
        required=False, allow_none=True, default=None, missing=None
    )
    profile_loss_id = fields.Str(
        required=False, allow_none=True, default=None, missing=None
    )


class CreateProjectOptimizationSchema(GetProjectOptimizationBestEstimatedResultsSchema):
    """
    Schema to use for creating a project optimization
    """

    name = fields.Str(required=False, default="", missing="")
    add_pruning = fields.Bool(required=False, default=True, missing=True)
    add_quantization = fields.Bool(required=False, default=False, missing=False)
    add_lr_schedule = fields.Bool(required=False, default=True, missing=True)
    add_trainable = fields.Bool(required=False, default=False, missing=False)


class UpdateProjectOptimizationSchema(Schema):
    """
    Schema to use for updating a project optimization
    """

    name = fields.Str(required=False)
    profile_perf_id = fields.Str(required=False, allow_none=True)
    profile_loss_id = fields.Str(required=False, allow_none=True)
    start_epoch = fields.Float(required=False)
    end_epoch = fields.Float(required=False)


class CreateUpdateProjectOptimizationModifiersPruningSchema(Schema):
    """
    Schema to use for creating or updating a project optimization pruning modifier
    """

    start_epoch = fields.Float(required=False)
    end_epoch = fields.Float(required=False)
    update_frequency = fields.Float(required=False)

    sparsity = fields.Float(required=False)
    balance_perf_loss = fields.Float(required=False, allow_none=False)
    filter_min_sparsity = fields.Float(required=False)
    filter_min_perf_gain = fields.Float(required=False)
    filter_min_recovery = fields.Float(required=False)

    nodes = fields.Nested(
        ProjectOptimizationModifierPruningNodeMetadataSchema, required=False, many=True,
    )


class CreateUpdateProjectOptimizationModifiersQuantizationSchema(Schema):
    """
    Schema to use for creating or updating a project optimization quantization modifier
    """

    start_epoch = fields.Float(required=False)
    end_epoch = fields.Float(required=False)

    level = fields.Str(required=False, validate=validate.OneOf(QUANTIZATION_LEVELS))
    balance_perf_loss = fields.Float(required=False, allow_none=False)
    filter_min_perf_gain = fields.Float(required=False)
    filter_min_recovery = fields.Float(required=False)
    nodes = fields.Nested(
        ProjectOptimizationModifierQuantizationNodeSchema, required=False, many=True
    )


class CreateUpdateProjectOptimizationModifiersLRScheduleSchema(Schema):
    """
    Schema to use for creating or updating a project optimization lr schedule modifier
    """

    lr_mods = fields.Nested(
        ProjectOptimizationModifierLRSchema, required=False, many=True,
    )


class CreateUpdateProjectOptimizationModifiersTrainableSchema(Schema):
    """
    Schema to use for creating or updating a project optimization trainable modifier
    """

    start_epoch = fields.Float(required=False)
    end_epoch = fields.Float(required=False)

    default_trainable = fields.Bool(required=False)

    nodes = fields.Nested(
        ProjectOptimizationModifierTrainableNodeSchema, required=False, many=True
    )


class SearchProjectOptimizationsSchema(Schema):
    """
    Schema to use for querying project optimizations
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


class ResponseProjectOptimizationFrameworksAvailableSchema(Schema):
    """
    Schema for returning the available frameworks for project optimization
    """

    frameworks = fields.List(
        fields.Str(validate=validate.OneOf(ML_FRAMEWORKS)), required=True
    )


class ResponseProjectOptimizationFrameworksAvailableSamplesSchema(Schema):
    """
    Schema for returning the available code samples for a framework
    for project optimization
    """

    framework = fields.Str(validate=validate.OneOf(ML_FRAMEWORKS), required=True)
    samples = fields.List(fields.Str(), required=True)


class ResponseProjectOptimizationModifiersAvailable(Schema):
    """
    Schema for returning the available modifiers for project optimization
    """

    modifiers = fields.List(
        fields.Str(validate=validate.OneOf(OPTIM_MODIFIER_TYPES)), required=True
    )


class ResponseProjectOptimizationModifiersBestEstimated(
    ProjectOptimizationModifierEstimationsSchema
):
    """
    Schema for returning the best estimated results for project optimization
    """

    pass


class ResponseProjectOptimizationSchema(Schema):
    """
    Schema for returning a project optimization
    """

    optim = fields.Nested(ProjectOptimizationSchema, required=True)


class ResponseProjectOptimizationsSchema(Schema):
    """
    Schema for returning multiple project optimizations
    """

    optims = fields.Nested(ProjectOptimizationSchema, required=True, many=True)


class ResponseProjectOptimizationDeletedSchema(Schema):
    """
    Schema for returning the results of deleting a project optimization
    """

    success = fields.Bool(required=False, default=True, missing=True)
    project_id = fields.Str(required=True)
    optim_id = fields.Str(required=True)


class ResponseProjectOptimizationModifierDeletedSchema(Schema):
    """
    Schema for returning the results of deleting a project optimization modifier
    """

    success = fields.Bool(required=False, default=True, missing=True)
    project_id = fields.Str(required=True)
    optim_id = fields.Str(required=True)
    modifer_id = fields.Str(required=True)
