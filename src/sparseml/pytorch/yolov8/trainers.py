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

import os
import shutil
import subprocess
import sys
import tempfile
from copy import deepcopy
from datetime import datetime

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import lr_scheduler

from ultralytics import __version__
from ultralytics.yolo.engine.trainer import BaseTrainer
from ultralytics.yolo.utils import DEFAULT_CONFIG
from ultralytics.yolo.utils.autobatch import check_train_batch_size
from ultralytics.yolo.utils.dist import (
    USER_CONFIG_DIR,
    ddp_cleanup,
    find_free_network_port,
)
from ultralytics.yolo.utils.torch_utils import ModelEMA, de_parallel, one_cycle
from ultralytics.yolo.v8.classify.train import ClassificationTrainer
from ultralytics.yolo.v8.detect.train import DetectionTrainer
from ultralytics.yolo.v8.segment.train import SegmentationTrainer


class SparseTrainer(BaseTrainer):
    def __init__(self, config=DEFAULT_CONFIG, overrides=None):
        super().__init__(config, overrides)

    def train(self):
        # NOTE: overriden to use our version of generate_ddp_command
        world_size = torch.cuda.device_count()
        if world_size > 1 and "LOCAL_RANK" not in os.environ:
            command = generate_ddp_command(world_size, self)
            try:
                subprocess.run(command)
            except Exception as e:
                self.console(e)
            finally:
                ddp_cleanup(command, self)
        else:
            self._do_train(int(os.getenv("RANK", -1)), world_size)

    def resume_training(self, ckpt):
        return super().resume_training(ckpt)

    def _setup_train(self, rank, world_size):
        return super()._setup_train(rank, world_size)

    def optimizer_step(self):
        return super().optimizer_step()

    def save_model(self):
        ckpt = {
            "epoch": self.epoch,
            "best_fitness": self.best_fitness,
            "model": deepcopy(de_parallel(self.model)).half(),
            "ema": deepcopy(self.ema.ema).half(),
            "updates": self.ema.updates,
            "optimizer": self.optimizer.state_dict(),
            "train_args": self.args,
            "date": datetime.now().isoformat(),
            "version": __version__,
        }

        # Save last, best and delete
        torch.save(ckpt, self.last)
        if self.best_fitness == self.fitness:
            torch.save(ckpt, self.best)
        del ckpt


class SparseDetectionTrainer(SparseTrainer, DetectionTrainer):
    pass


class SparseClassificationTrainer(SparseTrainer, ClassificationTrainer):
    pass


class SparseSegmentationTrainer(SparseTrainer, SegmentationTrainer):
    pass


def generate_ddp_command(world_size, trainer):
    import __main__  # noqa local import to avoid https://github.com/Lightning-AI/lightning/issues/15218

    file_name = os.path.abspath(sys.argv[0])
    using_cli = not file_name.endswith(".py")
    if using_cli:
        file_name = generate_ddp_file(trainer)
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        f"{world_size}",
        "--master_port",
        f"{find_free_network_port()}",
        file_name,
    ] + sys.argv[1:]


def generate_ddp_file(trainer):
    import_path = ".".join(str(trainer.__class__).split(".")[1:-1])

    if not trainer.resume:
        shutil.rmtree(trainer.save_dir)  # remove the save_dir
    content = f"""config = {dict(trainer.args)} \nif __name__ == "__main__":
    from ultralytics.{import_path} import {trainer.__class__.__name__}

    trainer = {trainer.__class__.__name__}(config=config)
    trainer.train()"""
    (USER_CONFIG_DIR / "DDP").mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix="_temp_",
        suffix=f"{id(trainer)}.py",
        mode="w+",
        encoding="utf-8",
        dir=USER_CONFIG_DIR / "DDP",
        delete=False,
    ) as file:
        file.write(content)
    return file.name
