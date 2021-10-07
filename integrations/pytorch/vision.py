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
Deprecated script kindly use the following task specific scripts for
 optimization tasks on image classification and object detection models:

* Model pruning - integrations/pytorch/train.py
* Quantization aware training - integrations/pytorch/train.py
* Sparse transfer learning - integrations/pytorch/train.py
* pruning sensitivity analysis - integrations/pytorch/pr_sensitivity.py
* learning rate sensitivity analysis - integrations/pytorch/lr_analysis.py
* ONNX export - integrations/pytorch/export.py
"""


if __name__ == "__main__":
    print(__doc__)
