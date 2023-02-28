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

"""
Definition of integration status table and CLI util to udpate status tables and
templates from yaml configs
"""

import os
from pathlib import Path

from pydantic import Field

from sparsezoo.utils.standardization import (
    FeatureStatus,
    FeatureStatusPage,
    FeatureStatusTable,
    write_status_pages,
)


class BaseTrainingStatusTable(FeatureStatusTable):

    cli: FeatureStatus = Field()
    api: FeatureStatus = Field()
    dense_training: FeatureStatus = Field()
    gradient_accumulation: FeatureStatus = Field()
    DP: FeatureStatus = Field()
    DDP: FeatureStatus = Field()

    @property
    def name(self) -> str:
        return "Base Training"


class SparsificationStatusTable(FeatureStatusTable):

    recipe: FeatureStatus = Field()
    recipe_args: FeatureStatus = Field()
    EMA: FeatureStatus = Field()
    AMP: FeatureStatus = Field()
    distillation: FeatureStatus = Field()

    @property
    def name(self) -> str:
        return "Sparsification"

    @property
    def description(self) -> str:
        return (
            "Features related to sparsification integration. "
            "Notes: \n"
            "* Recipe support should be optional\n"
            "* AMP must be disabled during QAT. (`scaler._enabled = False`)\n"
            "Distillation:\n"
            "* distillation_teacher kwarg must be passed to manager initialzation\n"
            "* Call loss = manager.loss_update(...) after loss is computed"
        )


class DatasetsStatusTable(FeatureStatusTable):

    use_standard_datasets: FeatureStatus = Field()
    train_val_test_datasets: FeatureStatus = Field()
    auto_download_datasets: FeatureStatus = Field()

    @property
    def name(self) -> str:
        return "Datasets"


class CheckpointsStatusTable(FeatureStatusTable):
    original_integration_checkpoints: FeatureStatus = Field()
    sparsezoo_checkpoints: FeatureStatus = Field()
    best_dense_checkpoint: FeatureStatus = Field()
    best_pruned_checkpoint: FeatureStatus = Field()
    best_pruned_quantized_checkpoint: FeatureStatus = Field()
    recipe_saved_to_checkpoint: FeatureStatus = Field()
    update_architecture_from_recipe: FeatureStatus = Field()
    staged_recipes: FeatureStatus = Field()

    @property
    def name(self) -> str:
        return "Checkpoints"

    @property
    def description(self) -> str:
        return (
            "Features related to checkpoints. "
            "Notes: \n"
            "* best_* checkpoints can only be saved after the entire sparsification "
            "step completes\n"
            "* update_architecture_from_recipe requires a call to apply_structure() "
            "on a torch model before loading sparsified checkpoint\n"
            "* staged_recipes requires manager.compose_staged(...) "
            "before checkpoint save"
        )


class LoggingStatusTable(FeatureStatusTable):
    stdout: FeatureStatus = Field()
    weights_and_biases: FeatureStatus = Field()
    tensorboard: FeatureStatus = Field()

    @property
    def name(self) -> str:
        return "Logging"

    @property
    def description(self) -> str:
        return (
            "Logging units for x axis in logging should be number of optimizer steps. "
            "Notably: `num_optimizer_steps = num_batches / gradient_accum_steps`. "
            "So when gradient_accumuluation is not used, the x axis will be number "
            "of batches trained on."
        )


class ExportStatusTable(FeatureStatusTable):
    cli: FeatureStatus = Field()
    api: FeatureStatus = Field()
    one_shot: FeatureStatus = Field()
    onnx: FeatureStatus = Field()
    torch_script: FeatureStatus = Field()
    static_batch_size: FeatureStatus = Field()
    dynamic_batch_size: FeatureStatus = Field()
    static_input_shape: FeatureStatus = Field()
    dynamic_input_shape: FeatureStatus = Field()
    save_to_simple_deployment_directory: FeatureStatus = Field()
    save_to_sparsezoo_directory: FeatureStatus = Field()

    @property
    def name(self) -> str:
        return "Export"

    @property
    def description(self) -> str:
        return (
            "PyTorch export features should use `ModuleExporter` and only require "
            "specifying checkpoint path and necessary configuration files"
        )


class SparseMLIntegrationStatusPage(FeatureStatusPage):
    base_training: BaseTrainingStatusTable = Field()
    sparsification: SparsificationStatusTable = Field()
    datasets: DatasetsStatusTable = Field()
    checkpoints: CheckpointsStatusTable = Field()
    logging: LoggingStatusTable = Field()
    export: ExportStatusTable = Field()

    @property
    def name(self) -> str:
        return "SparseML Integration Project"

    @property
    def description(self) -> str:
        return (
            "Feature status tables related to required and target features "
            "for SparseML sparsification aware training integrations"
        )


if __name__ == "__main__":
    status_dir = Path(__file__).parent.resolve()
    src_dir = os.path.join(Path(__file__).parent.parent.resolve(), "src")

    main_status_page_path = os.path.join(status_dir, "STATUS.MD")
    yaml_template_path = os.path.join(status_dir, "status_template.status.yaml")

    write_status_pages(
        status_page_class=SparseMLIntegrationStatusPage,
        root_directory=src_dir,
        main_status_page_path=main_status_page_path,
        yaml_template_path=yaml_template_path,
    )
