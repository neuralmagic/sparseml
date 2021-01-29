# Copyright 2021-present Neuralmagic, Inc.
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
Modifiers related to controlling the training epochs while training a model
"""

from sparseml.pytorch.optim.modifier import PyTorchModifierYAML, ScheduledModifier


__all__ = ["EpochRangeModifier"]


@PyTorchModifierYAML()
class EpochRangeModifier(ScheduledModifier):
    """
    Simple modifier to set the range of epochs for running in a scheduled optimizer
    (ie to set min and max epochs within a range without hacking other modifiers).

    Note, that if other modifiers exceed the range of this one for min or max epochs,
    this modifier will not have an effect.

    | Sample yaml:
    |   !EpochRangeModifier:
    |       start_epoch: 0
    |       end_epoch: 90

    :param start_epoch: The epoch to start the modifier at
    :param end_epoch: The epoch to end the modifier at
    """

    def __init__(
        self,
        start_epoch: float,
        end_epoch: float,
    ):
        super().__init__(
            start_epoch=start_epoch, end_epoch=end_epoch, end_comparator=-1
        )
