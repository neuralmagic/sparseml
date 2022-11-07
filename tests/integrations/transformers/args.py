# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional

from pydantic import BaseModel, Field

from sparseml.transformers.masked_language_modeling import MODEL_TYPES
from sparseml.transformers.text_classification import _TASK_TO_KEYS


class _TransformersTrainArgs(BaseModel):
    model_name_or_path: str = Field(
        description=(
            "Path to pretrained model, sparsezoo stub. or model identifier from "
            "huggingface.co/models"
        )
    )
    distill_teacher: Optional[str] = Field(
        default=None,
        description=("Teacher model which needs to be a trained NER model"),
    )
    config_name: Optional[str] = Field(
        default=None,
        description=("Pretrained config name or path if not the same as model_name"),
    )
    tokenizer_name: Optional[str] = Field(
        default=None,
        description=("Pretrained tokenizer name or path if not the same as model_name"),
    )
    cache_dir: Optional[str] = Field(
        default=None,
        description=("Where to store the pretrained data from huggingface.co"),
    )
    model_revision: str = Field(
        default="main",
        description=(
            "The specific model version to use "
            "(can be a branch name, tag name or commit id)"
        ),
    )
    use_auth_token: bool = Field(
        default=False,
        description=(
            "Will use token generated when running `transformers-cli login` "
            "(necessary to use this script with private models)"
        ),
    )
    recipe: Optional[str] = Field(
        default=None,
        description=(
            "Path to a SparseML sparsification recipe, see "
            "https://github.com/neuralmagic/sparseml for more information"
        ),
    )
    recipe_args: Optional[str] = Field(
        default=None,
        description="Recipe arguments to be overwritten",
    )
    dataset_name: Optional[str] = Field(
        default=None,
        description=("The name of the dataset to use (via the datasets library)"),
    )
    dataset_config_name: Optional[str] = Field(
        default=None,
        description=("The configuration name of the dataset to use"),
    )
    train_file: Optional[str] = Field(
        default=None,
        description=("A csv or a json file containing the training data."),
    )
    validation_file: Optional[str] = Field(
        default=None,
        description=("A csv or a json file containing the validation data."),
    )
    test_file: Optional[str] = Field(
        default=None,
        description=("A csv or a json file containing the test data."),
    )
    text_column_name: Optional[str] = Field(
        default=None,
        description=(
            "The column name of text to input in the file " "(a csv or JSON file)."
        ),
    )
    label_column_name: Optional[str] = Field(
        default=None,
        description=(
            "The column name of label to input in the file " "(a csv or JSON file)."
        ),
    )
    overwrite_cache: bool = Field(
        default=False,
        description=("Overwrite the cached training and evaluation sets"),
    )
    preprocessing_num_workers: Optional[int] = Field(
        default=None,
        description=("The number of processes to use for the preprocessing."),
    )
    max_seq_length: int = Field(
        default=None,
        description=(
            "The maximum total input sequence length after tokenization. "
            "If set, sequences longer than this will be truncated, sequences shorter "
            "will be padded."
        ),
    )
    pad_to_max_length: bool = Field(
        default=False,
        description=(
            "Whether to pad all samples to `max_seq_length`. If False, "
            "will pad the samples dynamically when batching to the maximum length "
            "in the batch (which can be faster on GPU but will be slower on TPU)."
        ),
    )
    max_train_samples: Optional[int] = Field(
        default=None,
        description=(
            "For debugging purposes or quicker training, truncate the number "
            "of training examples to this value if set."
        ),
    )
    max_eval_samples: Optional[int] = Field(
        default=None,
        description=(
            "For debugging purposes or quicker training, truncate the number "
            "of evaluation examples to this value if set."
        ),
    )
    one_shot: bool = Field(
        default=False,
        description="Whether to apply recipe in a one shot manner.",
    )
    output_dir: str = Field(
        default=".",
        description="The output directory where the model predictions and checkpoints "
        " will be written.",
    )
    overwrite_output_dir: bool = Field(
        default=False,
        description=(
            "Overwrite the content of the output directory. "
            "Use this to continue training if output_dir points to a checkpoint "
            "directory."
        ),
    )

    do_train: bool = Field(default=False, description="Whether to run training.")
    do_eval: bool = Field(
        default=False, description="Whether to run eval on the dev set."
    )
    do_predict: bool = Field(
        default=False, description="Whether to run predictions on the test set."
    )
    evaluation_strategy: str = Field(
        default="no", description="The evaluation strategy to use."
    )
    prediction_loss_only: bool = Field(
        default=False,
        description="When performing evaluation and predictions, only returns the "
        "loss.",
    )

    per_device_train_batch_size: int = Field(
        default=8, description="Batch size per GPU/TPU core/CPU for training."
    )
    per_device_eval_batch_size: int = Field(
        default=8, description="Batch size per GPU/TPU core/CPU for evaluation."
    )

    per_gpu_train_batch_size: Optional[int] = Field(
        default=None,
        description="Deprecated, the use of `--per_device_train_batch_size` is "
        "preferred. Batch size per GPU/TPU core/CPU for training.",
    )
    per_gpu_eval_batch_size: Optional[int] = Field(
        default=None,
        description="Deprecated, the use of `--per_device_eval_batch_size` is "
        "preferred. Batch size per GPU/TPU core/CPU for evaluation.",
    )

    gradient_accumulation_steps: int = Field(
        default=1,
        description="Number of updates steps to accumulate before performing a "
        "backward/update pass.",
    )
    eval_accumulation_steps: Optional[int] = Field(
        default=None,
        description="Number of predictions steps to accumulate before moving the "
        "tensors to the CPU.",
    )

    learning_rate: float = Field(
        default=5e-5, description="The initial learning rate for AdamW."
    )
    weight_decay: float = Field(
        default=0.0, description="Weight decay for AdamW if we apply some."
    )
    adam_beta1: float = Field(default=0.9, description="Beta1 for AdamW optimizer")
    adam_beta2: float = Field(default=0.999, description="Beta2 for AdamW optimizer")
    adam_epsilon: float = Field(
        default=1e-8, description="Epsilon for AdamW optimizer."
    )
    max_grad_norm: float = Field(default=1.0, description="Max gradient norm.")

    num_train_epochs: Optional[float] = Field(
        default=None, description="Total number of training epochs to perform."
    )
    max_steps: int = Field(
        default=-1,
        description="If > 0: set total number of training steps to perform. "
        "Override num_train_epochs.",
    )
    lr_scheduler_type: str = Field(
        default="linear", description="The scheduler type to use."
    )
    warmup_ratio: float = Field(
        default=0.0,
        description="Linear warmup over warmup_ratio fraction of total steps.",
    )
    warmup_steps: int = Field(default=0, description="Linear warmup over warmup_steps.")

    log_level: Optional[str] = Field(
        default="passive",
        description="Logger log level to use on the main node. Possible choices are "
        "the log levels as strings: 'debug', 'info', 'warning', 'error' and "
        "'critical', plus a 'passive' level which doesn't set anything and lets the "
        "application set the level. Defaults to 'passive'.",
    )
    log_level_replica: Optional[str] = Field(
        default="passive",
        description="Logger log level to use on replica nodes. Same choices and "
        "defaults as ``log_level``",
    )
    log_on_each_node: bool = Field(
        default=True,
        description="When doing a multinode distributed training, whether to log "
        "once per node or just once on the main node.",
    )
    logging_dir: Optional[str] = Field(default=None, description="Tensorboard log dir.")
    logging_strategy: str = Field(
        default="steps", description="The logging strategy to use."
    )
    logging_first_step: bool = Field(
        default=False, description="Log the first global_step"
    )
    logging_steps: int = Field(default=500, description="Log every X updates steps.")
    """
    Needs to be updated to type(bool) in transformers repo
    logging_nan_inf_filter: bool = Field(
        default=True, description="Filter nan and inf losses for logging."
    )
    """
    save_strategy: str = Field(
        default="steps", description="The checkpoint save strategy to use."
    )
    save_steps: int = Field(
        default=500, description="Save checkpoint every X updates steps."
    )
    save_total_limit: Optional[int] = Field(
        default=None,
        description=(
            "Limit the total amount of checkpoints. "
            "Deletes the older checkpoints in the output_dir. Default is unlimited "
            "checkpoints"
        ),
    )
    save_on_each_node: bool = Field(
        default=False,
        description="When doing multi-node distributed training, whether to save "
        "models and checkpoints on each node, or only on the main one",
    )
    no_cuda: bool = Field(
        default=False, description="Do not use CUDA even when it is available"
    )
    seed: int = Field(
        default=42,
        description="Random seed that will be set at the beginning of training.",
    )
    data_seed: int = Field(
        default=None, description="Random seed to be used with data samplers."
    )
    bf16: bool = Field(
        default=False,
        description="Whether to use bf16 (mixed) precision instead of 32-bit. "
        "Requires Ampere or higher NVIDIA architecture. This is an experimental API "
        "and it may change.",
    )
    fp16: bool = Field(
        default=False,
        description="Whether to use fp16 (mixed) precision instead of 32-bit",
    )
    fp16_opt_level: str = Field(
        default="O1",
        description=(
            "For fp16: Apex AMP optimization level selected in "
            "['O0', 'O1', 'O2', and 'O3']. "
            "See details at https://nvidia.github.io/apex/amp.html"
        ),
    )
    half_precision_backend: str = Field(
        default="auto", description="The backend to be used for half precision."
    )
    bf16_full_eval: bool = Field(
        default=False,
        description="Whether to use full bfloat16 evaluation instead of 32-bit. "
        "This is an experimental API and it may change.",
    )
    fp16_full_eval: bool = Field(
        default=False,
        description="Whether to use full float16 evaluation instead of 32-bit",
    )
    tf32: bool = Field(
        default=None,
        description="Whether to enable tf32 mode, available in Ampere and newer "
        "GPU architectures. This is an experimental API and it may change.",
    )
    local_rank: int = Field(
        default=-1, description="For distributed training: local_rank"
    )
    xpu_backend: str = Field(
        default=None,
        description="The backend to be used for distributed training on Intel XPU.",
    )
    tpu_num_cores: Optional[int] = Field(
        default=None,
        description="TPU: Number of TPU cores (automatically passed by launcher "
        "script)",
    )
    tpu_metrics_debug: bool = Field(
        default=False,
        description="Deprecated, the use of `--debug tpu_metrics_debug` is preferred. "
        "TPU: Whether to print debug metrics",
    )
    debug: Optional[str] = Field(
        default=None,
        description="Whether or not to enable debug mode. Current options: "
        "`underflow_overflow` (Detect underflow and overflow in activations and "
        "weights), `tpu_metrics_debug` (print debug metrics on TPU).",
    )

    dataloader_drop_last: bool = Field(
        default=False,
        description="Drop the last incomplete batch if it is not divisible by the "
        "batch size.",
    )
    eval_steps: int = Field(
        default=None, description="Run an evaluation every X steps."
    )
    dataloader_num_workers: int = Field(
        default=0,
        description="Number of subprocesses to use for data loading (PyTorch only). "
        "0 means that the data will be loaded in the main process.",
    )

    past_index: int = Field(
        default=-1,
        description="If >=0, uses the corresponding part of the output as the past "
        "state for next step.",
    )

    run_name: Optional[str] = Field(
        default=None,
        description="An optional descriptor for the run. Notably used for "
        "wandb logging.",
    )
    disable_tqdm: Optional[bool] = Field(
        default=None, description="Whether or not to disable the tqdm progress bars."
    )

    remove_unused_columns: Optional[bool] = Field(
        default=True,
        description="Remove columns not required by the model when using an "
        "nlp.Dataset.",
    )
    label_names: Optional[List[str]] = Field(
        default=None,
        description="The list of keys in your dictionary of inputs that correspond "
        "to the labels.",
    )

    load_best_model_at_end: Optional[bool] = Field(
        default=False,
        description="Whether or not to load the best model found during training "
        "at the end of training.",
    )
    metric_for_best_model: Optional[str] = Field(
        default=None, description="The metric to use to compare two different models."
    )
    greater_is_better: Optional[bool] = Field(
        default=None,
        description="Whether the `metric_for_best_model` should be maximized or not.",
    )
    ignore_data_skip: bool = Field(
        default=False,
        description="When resuming training, whether or not to skip the first epochs "
        "and batches to get to the same training data.",
    )
    sharded_ddp: Optional[str] = Field(
        default=None,
        description="Whether or not to use sharded DDP training (in distributed "
        "training only). The base option should be `simple`, `zero_dp_2` or "
        "`zero_dp_3` and you can add CPU-offload to `zero_dp_2` or `zero_dp_3` like "
        "this: zero_dp_2 offload` or `zero_dp_3 offload`. You can add auto-wrap to "
        "`zero_dp_2` or with the same syntax: zero_dp_2 auto_wrap` or "
        "`zero_dp_3 auto_wrap`.",
    )
    deepspeed: Optional[str] = Field(
        default=None,
        description="Enable deepspeed and pass the path to deepspeed json config file "
        "(e.g. ds_config.json) or an already loaded json file as a dict",
    )
    label_smoothing_factor: float = Field(
        default=0.0,
        description="The label smoothing epsilon to apply (zero means no "
        "label smoothing).",
    )
    optim: str = Field(default="adamw_hf", description="The optimizer to use.")
    adafactor: bool = Field(
        default=False, description="Whether or not to replace AdamW by Adafactor."
    )
    group_by_length: bool = Field(
        default=False,
        description="Whether or not to group samples of roughly the same length "
        "together when batching.",
    )
    length_column_name: Optional[str] = Field(
        default="length",
        description="Column name with precomputed lengths to use when grouping by "
        "length.",
    )
    report_to: Optional[List[str]] = Field(
        default=None,
        description="The list of integrations to report the results and logs to.",
    )
    ddp_find_unused_parameters: Optional[bool] = Field(
        default=None,
        description="When using distributed training, the value of the flag "
        "`find_unused_parameters` passed to `DistributedDataParallel`.",
    )
    ddp_bucket_cap_mb: Optional[int] = Field(
        default=None,
        description="When using distributed training, the value of the flag "
        "`bucket_cap_mb` passed to `DistributedDataParallel`.",
    )
    dataloader_pin_memory: bool = Field(
        default=True, description="Whether or not to pin memory for DataLoader."
    )
    skip_memory_metrics: bool = Field(
        default=True,
        description="Whether or not to skip adding of memory profiler reports to "
        "metrics.",
    )
    use_legacy_prediction_loop: bool = Field(
        default=False,
        description="Whether or not to use the legacy prediction_loop in the Trainer.",
    )
    push_to_hub: bool = Field(
        default=False,
        description="Whether or not to upload the trained model to the model hub after "
        "training.",
    )
    resume_from_checkpoint: Optional[str] = Field(
        default=None,
        description="The path to a folder with a valid checkpoint for your model.",
    )
    hub_model_id: str = Field(
        default=None,
        description="The name of the repository to keep in sync with the local "
        "`output_dir`.",
    )
    hub_strategy: str = Field(
        default="every_save",
        description="The hub strategy to use when `--push_to_hub` is activated.",
    )
    hub_token: str = Field(
        default=None, description="The token to use to push to the Model Hub."
    )
    gradient_checkpointing: bool = Field(
        default=False,
        description="If True, use gradient checkpointing to save memory at the expense "
        "of slower backward pass.",
    )
    # Deprecated arguments
    fp16_backend: str = Field(
        default="auto", description="Deprecated. Use half_precision_backend instead"
    )
    push_to_hub_model_id: str = Field(
        default=None,
        description="The name of the repository to which push the `Trainer`.",
    )
    push_to_hub_organization: str = Field(
        default=None,
        description="The name of the organization in with to which push the `Trainer`.",
    )
    push_to_hub_token: str = Field(
        default=None, description="The token to use to push to the Model Hub."
    )
    _n_gpu: int = Field(init=False, repr=False, default=-1)
    mp_parameters: Optional[str] = Field(
        default=None,
        description="Used by the SageMaker launcher to send mp-specific args. "
        "Ignored in Trainer",
    )


class QuestionAnsweringArgs(_TransformersTrainArgs):
    max_predict_samples: Optional[int] = Field(
        default=None,
        description=(
            "For debugging purposes or quicker training, truncate the number of "
            "prediction examples to this value if set."
        ),
    )
    version_2_with_negative: bool = Field(
        default=False,
        description=("If true, some of the examples do not have an answer."),
    )
    null_score_diff_threshold: float = Field(
        default=0.0,
        description=(
            "The threshold used to select the null answer: if the best answer has "
            "a score that is less than the score of the null answer minus this "
            "threshold, the null answer is selected for this example. Only useful "
            "when `version_2_with_negative=True`."
        ),
    )
    doc_stride: int = Field(
        default=128,
        description=(
            "When splitting up a long document into chunks, how much stride to "
            "take between chunks."
        ),
    )
    n_best_size: int = Field(
        default=20,
        description=(
            "The total number of n-best predictions to generate when looking "
            "for an answer."
        ),
    )
    max_answer_length: int = Field(
        default=30,
        description=(
            "The maximum length of an answer that can be generated. This is "
            "needed because the start and end predictions are not conditioned "
            "on one another."
        ),
    )


class TextClassificationArgs(_TransformersTrainArgs):
    max_predict_samples: Optional[int] = Field(
        default=None,
        description=(
            "For debugging purposes or quicker training, truncate the number of "
            "prediction examples to this value if set."
        ),
    )
    task_name: Optional[str] = Field(
        default=None,
        description=(
            "The name of the task to train on: " + ", ".join(_TASK_TO_KEYS.keys())
        ),
    )


class TokenClassificationArgs(_TransformersTrainArgs):
    max_predict_samples: Optional[int] = Field(
        default=None,
        description=(
            "For debugging purposes or quicker training, truncate the number of "
            "prediction examples to this value if set."
        ),
    )
    label_all_tokens: bool = Field(
        default=False,
        description=(
            "Whether to put the label for one word on all tokens of generated "
            "by that word or just on the "
            "one (in which case the other tokens will have a padding index)."
        ),
    )
    return_entity_level_metrics: bool = Field(
        default=False,
        description=(
            "Whether to return all the entity levels during evaluation or "
            "just the overall ones."
        ),
    )
    task_name: Optional[str] = Field(
        default="ner", description=("The name of the task (ner, pos...).")
    )


class MaskedLanguageModellingArgs(_TransformersTrainArgs):
    task_name: Optional[str] = Field(
        default=None,
        description=(
            "The name of the task to train on: " + ", ".join(_TASK_TO_KEYS.keys())
        ),
    )

    dataset_name_2: Optional[str] = Field(
        default=None,
        description=("The name of the dataset to use (via the datasets library)"),
    )
    dataset_config_name_2: Optional[str] = Field(
        default=None,
        description=("The configuration name of the dataset to use"),
    )
    validation_split_percentage: Optional[int] = Field(
        default=5,
        description=(
            "The percentage of the train set used as validation set in case "
            "there's no validation split"
        ),
    )
    mlm_probability: float = Field(
        default=0.15,
        description=("Ratio of tokens to mask for masked language modeling loss"),
    )
    line_by_line: bool = Field(
        default=False,
        description=(
            "Whether distinct lines of text in the dataset are to be handled as "
            "distinct sequences."
        ),
    )


class TransformersExportArgs(BaseModel):
    model_type: Optional[str] = Field(
        default=None,
        description=(
            "If training from scratch, pass a model type from the list: "
            + ", ".join(MODEL_TYPES)
        ),
    )
    task: str = Field(
        description="Task to create the model for. i.e. mlm, qa, glue, ner"
    )
    model_path: Optional[str] = Field(
        default=None,
        description=(
            "Path to directory where model files for weights, config, and "
            "tokenizer are stored"
        ),
    )
    sequence_length: int = Field(
        default=384,
        description="Sequence length to use. Default is 384. Can be overwritten later",
    )
    no_convert_qat: bool = Field(
        default=True,
        description=(
            "Set flag to not perform QAT to fully quantized conversion after export"
        ),
    )
    finetuning_task: Optional[str] = Field(
        default=None,
        description=(
            "Optional finetuning task for text classification and token "
            "classification exports"
        ),
    )
    onnx_file_name: str = Field(
        default="model.onnx",
        description=(
            "Name for exported ONNX file in the model directory. "
            "Default and reccomended value for pipeline compatibility is 'model.onnx'"
        ),
    )


class TransformersDeployArgs(BaseModel):
    task: str = Field(description="Task to create the model for.")
    model_path: Optional[str] = Field(
        default=None,
        description=("Path to directory where model onnx file is stored"),
    )
