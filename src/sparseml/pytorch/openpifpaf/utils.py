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

import argparse
import hashlib
import logging
import shutil

import torch

import openpifpaf
from sparseml.pytorch.optim import ScheduledModifierManager


LOG = logging.getLogger(__name__)


class SparseMLTrainer(openpifpaf.network.Trainer):
    recipe: str = None

    def __init__(
        self,
        model: torch.nn.Module,
        loss,
        optimizer,
        out,
        num_batches_per_epoch,
        *,
        checkpoint_shell=None,
        lr_scheduler=None,
        device=None,
        model_meta_data=None,
    ):
        self.manager = ScheduledModifierManager.from_yaml(SparseMLTrainer.recipe)

        self.epochs = self.manager.max_epochs

        if self.manager.learning_rate_modifiers:
            lr_scheduler = None

        self.manager.initialize(model)
        optimizer = self.manager.modify(model, optimizer, num_batches_per_epoch)

        super().__init__(
            model,
            loss,
            optimizer,
            out,
            checkpoint_shell=checkpoint_shell,
            lr_scheduler=lr_scheduler,
            device=device,
            model_meta_data=model_meta_data,
        )

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        openpifpaf.network.Trainer.cli(parser)
        group = parser.add_argument_group("sparseml")
        group.add_argument(
            "--recipe", default=None, required=True, help="Path to sparseml recipe"
        )

    @classmethod
    def configure(cls, args: argparse.Namespace):
        openpifpaf.network.Trainer.configure(args)
        cls.recipe = args.recipe

    def loop(
        self,
        train_scenes: torch.utils.data.DataLoader,
        val_scenes: torch.utils.data.DataLoader,
        start_epoch=0,
    ):
        super().loop(train_scenes, val_scenes, start_epoch)
        self.manager.finalize(self.model)

    def train(self, scenes, epoch):
        if self.manager.qat_active(epoch=epoch):
            self.ema_restore_params = None
        return super().train(scenes, epoch)

    def write_model(self, epoch, final=True):
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return

        model_to_save = self.model
        if self.checkpoint_shell is not None:
            model = (
                self.model if not hasattr(self.model, "module") else self.model.module
            )
            self.checkpoint_shell.load_state_dict(model.state_dict())
            model_to_save = self.checkpoint_shell

        filename = "{}.epoch{:03d}".format(self.out, epoch)
        LOG.debug("about to write model")
        torch.save(
            {
                "model": model_to_save,
                "state_dict": model_to_save.state_dict(),
                "checkpoint_recipe": str(self.manager),
                "epoch": epoch,
                "meta": self.model_meta_data,
            },
            filename,
        )
        LOG.info("model written: %s", filename)

        if final:
            sha256_hash = hashlib.sha256()
            with open(filename, "rb") as f:
                for byte_block in iter(lambda: f.read(8192), b""):
                    sha256_hash.update(byte_block)
            file_hash = sha256_hash.hexdigest()
            outname, _, outext = self.out.rpartition(".")
            final_filename = "{}-{}.{}".format(outname, file_hash[:8], outext)
            shutil.copyfile(filename, final_filename)
