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

from sparseml.optim import BaseManager, BaseScheduled


def test_manager():
    manager = BaseManager(
        modifiers=[
            BaseScheduled(
                start_epoch=1.0,
                min_start=0,
                end_epoch=2.0,
                min_end=0,
                end_comparator=-1,
            ),
            BaseScheduled(
                start_epoch=5.0,
                min_start=0,
                end_epoch=10.0,
                min_end=0,
                end_comparator=-1,
            ),
        ]
    )
    assert manager.min_epochs == 1.0
    assert manager.max_epochs == 10.0
