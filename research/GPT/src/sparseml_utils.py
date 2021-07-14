import logging
import collections
import math
import time
import json
import os
import numpy as np
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union


from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, IterableDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler

import pdb
import torch
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

from sparseml.pytorch.optim.manager import ScheduledModifierManager
from sparseml.pytorch.optim.optimizer import ScheduledOptimizer
from sparseml.pytorch.utils import ModuleExporter, logger

from transformers import Trainer
from transformers.trainer_utils import PredictionOutput, EvalLoopOutput
from transformers.file_utils import is_torch_tpu_available
from transformers.trainer_pt_utils import find_batch_size

logger = logging.getLogger(__name__)


class SparseMLTrainer(Trainer):
    """
    Question Answering trainer with SparseML integration

    :param recipe: recipe for model sparsification
    :param teacher: teacher model for distillation
    :param distill_hardness: ratio of loss by teacher targets (between 0 and 1)
    :param distill_temperature: temperature for distillation
    :param args, kwargs: arguments passed into parent class
    """

    def __init__(self, recipe, prediction_file=None, teacher=None, distill_hardness=0.5, distill_temperature=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.recipe = recipe
        self.teacher = teacher
        self.distill_hardness = distill_hardness
        self.distill_temperature = distill_temperature
        self.criterion = torch.nn.CrossEntropyLoss()
        self.manager = None
        self.loggers = None
        if self.recipe is not None:
            loggers = []
            self.loggers = loggers

    def create_optimizer(self):
        """
        Create optimizer customized using SparseML
        """
        super().create_optimizer()
        if self.recipe is None:
            return
        steps_per_epoch = math.ceil(
            len(self.train_dataset) / (self.args.per_device_train_batch_size * self.args._n_gpu)
        )
        self.manager = ScheduledModifierManager.from_yaml(self.recipe)
        self.args.num_train_epochs = float(self.manager.max_epochs)
        if hasattr(self, "scaler"):
            self.manager.initialize(self.model, epoch=0.0, loggers=self.loggers)
            self.scaler = self.manager.modify(
                self.model, self.optimizer, steps_per_epoch=steps_per_epoch, wrap_optim=self.scaler
            )
        else:
            self.optimizer = ScheduledOptimizer(
                self.optimizer, self.model, self.manager, steps_per_epoch=steps_per_epoch, loggers=self.loggers
            )

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Computing loss using teacher/student distillation
        """
        if self.recipe is None or self.teacher is None:
            return super().compute_loss(model, inputs, return_outputs=return_outputs)
        outputs = model(**inputs)
        if self.teacher is None:
            loss = outputs["loss"]
        else:
            input_device = inputs["input_ids"].device
            self.teacher = self.teacher.to(input_device)
            logits_student = outputs["logits"]
            logits_label = inputs["labels"]
            with torch.no_grad():
                teacher_output = self.teacher(**inputs)
            #pdb.set_trace()
            logits_teacher = teacher_output["logits"]
            teacher_loss = (
                F.kl_div(input=F.log_softmax(logits_student / self.distill_temperature, dim=-1),target=F.softmax(logits_teacher / self.distill_temperature, dim=-1),reduction="batchmean",) * (self.distill_temperature ** 2)
            )
            label_loss = outputs["loss"]
            loss = ((1 - self.distill_hardness) * label_loss) + (self.distill_hardness * teacher_loss)
        return (loss, outputs) if return_outputs else loss

def export_model(model, dataloader, output_dir):
    """
    Export a trained model to ONNX
    :param model: trained model
    :param dataloader: dataloader to get sample batch
    :param output_dir: output directory for ONNX model
    """
    exporter = ModuleExporter(model, output_dir=output_dir)
    for _, sample_batch in enumerate(dataloader):
        sample_input = (sample_batch["input_ids"], sample_batch["attention_mask"], sample_batch["token_type_ids"])
        exporter.export_onnx(sample_batch=sample_input, convert_qat=True)
        break