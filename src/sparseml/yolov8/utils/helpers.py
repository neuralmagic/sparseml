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
import warnings
from argparse import Namespace


__all__ = ["check_coco128_segmentation"]


def check_coco128_segmentation(args: Namespace) -> Namespace:
    """
    Checks if the argument 'data' is coco128.yaml and if so,
    replaces it with coco128-seg.yaml.
    :param args: arguments to check
    :return: the updated arguments
    """
    if args.data == "coco128.yaml":
        dataset_name, dataset_extension = os.path.splitext(args.data)
        dataset_yaml = dataset_name + "-seg" + dataset_extension
        warnings.warn(
            f"Dataset yaml {args.data} is not supported for segmentation. "
            f"Attempting to use {dataset_yaml} instead."
        )
        args.data = dataset_yaml
    return args
