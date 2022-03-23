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

import platform
import warnings
from collections import OrderedDict

import torch


__all__ = ["MetadataManager"]

QUESTION_ANSWERING_METADATA = [
    "distill_hardness",
    "distill_temperature",
    "per_device_train_batch_size",
    "learning_rate",
    "max_seq_length",
    "doc_stride",
    "num_train_epochs",
    "warmup_steps",
    "fp16",
]

MASKED_LANGUAGE_MODELING_METADATA = [
    "dataset_name",
    "dataset_name_2",
    "dataset_config_name_2",
    "per_device_train_batch_size",
    "per_device_eval_batch_size",
    "fp16",
    "learning_rate",
    "warmup_steps",
    "max_seq_length",
    "num_train_epochs",
]


class MetadataManager:
    """
    Helper class to manage metadata in SparseML pipelines.
    """

    def __init__(self, task):
        supported_tasks = [
            "question_answering",
            "masked_language_modeling",
            "text_classification",
            "token_classification",
        ]
        if task not in supported_tasks:
            raise ValueError(
                f"Passed task {task} but MetadataManager "
                f"supports only {supported_tasks}"
            )
        elif task == "question_answering":
            self.training_args_metadata = QUESTION_ANSWERING_METADATA
        elif task == "masked_language_modeling":
            self.training_args_metadata = MASKED_LANGUAGE_MODELING_METADATA
        else:
            raise NotImplementedError("The chosen task is not supported yet.")

        self.task = task
        self._metadata = OrderedDict()
        self._metadata["python_version"] = platform.python_version()
        self._metadata["sparseml_version"] = torch.__version__

        self._metadata.update(
            OrderedDict([(k, None) for k in self.training_args_metadata])
        )

    @property
    def metadata(self):
        """
        Implementational decision: when fetching metadata to pass to a Manager,
        what to do with entries in self._metadata which have None keys?
        1. We remove them to declutter the final saved staged recipe?
        2. We enforce strict behavior and raise ValueError if any key has None key
            (better: raise warning for the user?).
        """
        warnings.warn(
            f"Detected metadata keys which are None."
            f"The task {self.task} expects the following "
            f"metadata entries to be non-None: "
            f"{[k for k,v in self.metadata.items() if v is None]}"
        )
        return OrderedDict([(k, v) for k, v in self._metadata.items() if v is not None])

    @metadata.setter
    def metadata(self, data):
        # expecting data to be a dictionary
        if not isinstance(data, dict):
            ValueError(
                "Expected type dict to update the metadata of the MetadataManager."
            )
        # including only the expected keys
        data = {k: v for k, v in data.items() if k in self._metadata.keys()}
        self._metadata.update(data)
