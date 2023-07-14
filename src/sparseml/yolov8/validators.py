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

import json
from argparse import Namespace
from pathlib import Path

import torch
from tqdm import tqdm

from sparseml.pytorch.utils import detach
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.yolo.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.yolo.engine.validator import BaseValidator
from ultralytics.yolo.utils import LOGGER, TQDM_BAR_FORMAT, callbacks, emojis
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.ops import Profile
from ultralytics.yolo.utils.torch_utils import de_parallel, select_device
from ultralytics.yolo.v8.classify.val import ClassificationValidator
from ultralytics.yolo.v8.detect.val import DetectionValidator
from ultralytics.yolo.v8.segment.val import SegmentationValidator


class SparseValidator(BaseValidator):
    def __call__(self, trainer=None, model=None, training=True):
        self.training = trainer is not None and training
        if self.training:
            self.device = trainer.device
            self.data = trainer.data
            if trainer.manager and trainer.manager.quantization_modifiers:
                # Since we disable the EMA model for QAT, we validate the non-averaged
                # QAT model
                model = de_parallel(trainer.model)
            else:
                model = trainer.ema.ema or trainer.model

            # self.args.half = self.device.type != "cpu"
            model = model.half() if self.args.half else model.float()
            self.model = model
            self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)

            self.args.plots = (
                trainer.stopper.possible_stop or trainer.epoch == trainer.epochs - 1
            )  # always plot final epoch

            model.eval()
        else:
            callbacks.add_integration_callbacks(self)
            self.run_callbacks("on_val_start")
            assert model is not None, "Either trainer or model is needed for validation"
            self.device = select_device(self.args.device, self.args.batch)
            # self.args.half &= self.device.type != "cpu"
            model = AutoBackend(
                model,
                device=self.device,
                dnn=self.args.dnn,
                data=self.args.data,
                fp16=self.args.half,
            )
            self.model = model
            stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
            imgsz = check_imgsz(self.args.imgsz, stride=stride)
            if engine:
                self.args.batch = model.batch_size
            else:
                self.device = model.device
                if not pt and not jit:
                    self.args.batch = 1  # export.py models default to batch-size 1
                    LOGGER.info(
                        "Forcing --batch-size 1 square inference (1,3,"
                        f"{imgsz},{imgsz}) for non-PyTorch models"
                    )

            if isinstance(self.args.data, str) and self.args.data.endswith(".yaml"):
                self.data = check_det_dataset(self.args.data)
            elif self.args.task == "classify":
                self.data = check_cls_dataset(self.args.data, split=self.args.split)
            else:
                raise FileNotFoundError(
                    emojis(f"Dataset '{self.args.data}' not found ❌")
                )
            if isinstance(self.data["path"], str):
                self.data["path"] = Path(self.data["path"])

            if self.device.type == "cpu":
                self.args.workers = (
                    0  # faster CPU val as time dominated by inference, not dataloading
                )
            if not pt:
                self.args.rect = False
            self.dataloader = self.dataloader or self.get_dataloader(
                self.data.get(self.args.split), self.args.batch
            )

            model.eval()
            model.warmup(
                imgsz=(1 if pt else self.args.batch, 3, imgsz, imgsz)
            )  # warmup
            trainer.model = model.model
            trainer.model.args = Namespace(**model.model.args)

        dt = Profile(), Profile(), Profile(), Profile()
        n_batches = len(self.dataloader)
        desc = self.get_desc()
        # NOTE: keeping `not self.training` in tqdm will eliminate pbar after
        # segmentation evaluation during training,
        # which may affect classification task since this arg
        # is in yolov5/classify/val.py.
        # bar = tqdm(self.dataloader, desc, n_batches, not self.training,
        # bar_format=TQDM_BAR_FORMAT)
        bar = tqdm(self.dataloader, desc, n_batches, bar_format=TQDM_BAR_FORMAT)
        self.init_metrics(de_parallel(model))
        self.jdict = []  # empty before each val
        for batch_i, batch in enumerate(bar):
            self.run_callbacks("on_val_batch_start")
            self.batch_i = batch_i
            # pre-process
            with dt[0]:
                batch = self.preprocess(batch)

            # inference
            with dt[1]:
                preds = model(batch["img"], augment=self.args.augment)

            # loss
            with dt[2]:
                if self.training:
                    self.loss += model.loss(batch=batch, preds=preds)[1]
                else:
                    if hasattr(self, "loss"):
                        self.loss += model.model.loss(batch=batch, preds=preds[1])[1]
                    else:
                        self.loss = model.model.loss(batch=batch, preds=preds[1])[1]

            # pre-process predictions
            with dt[3]:
                preds = self.postprocess(preds)

            # During QAT the resulting preds are grad required, breaking
            # the update metrics function.
            detached_preds = detach(preds)
            self.update_metrics(detached_preds, batch)

            if self.args.plots and batch_i < 3:
                self.plot_val_samples(batch, batch_i)
                self.plot_predictions(batch, detached_preds, batch_i)

            self.run_callbacks("on_val_batch_end")
        stats = self.get_stats()
        self.check_stats(stats)
        self.speed = dict(
            zip(
                self.speed.keys(),
                (x.t / len(self.dataloader.dataset) * 1e3 for x in dt),
            )
        )
        self.finalize_metrics()
        self.run_callbacks("on_val_end")
        model.float()
        results = {
            **stats,
            **trainer.label_loss_items(
                self.loss.cpu() / len(self.dataloader), prefix="val"
            ),
        }
        if self.training:
            return {
                k: round(float(v), 5) for k, v in results.items()
            }  # return results as 5 decimal place floats
        else:
            LOGGER.info(
                "Speed: %.1fms preprocess, %.1fms inference, %.1fms loss, %.1fms "
                "postprocess per image" % tuple(self.speed.values())
            )
            LOGGER.info(f"Validation loss: {results['val/Loss']}")
            LOGGER.info(f"Metrics: {results}")
            if self.args.save_json and self.jdict:
                with open(str(self.save_dir / "predictions.json"), "w") as f:
                    LOGGER.info(f"Saving {f.name}...")
                    json.dump(self.jdict, f)  # flatten and save
                stats = self.eval_json(stats)  # update stats
            if self.args.plots or self.args.save_json:
                LOGGER.info(f"Results saved to {self.save_dir}")
            return stats


class SparseDetectionValidator(SparseValidator, DetectionValidator):
    ...


class SparseClassificationValidator(SparseValidator, ClassificationValidator):
    ...


class SparseSegmentationValidator(SparseValidator, SegmentationValidator):
    ...
