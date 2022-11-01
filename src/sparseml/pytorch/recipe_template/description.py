# flake8: noqa W293, W291

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

__all__ = ["DESCRIPTION"]

DESCRIPTION: str = """
This recipe defines the hyperparams necessary to apply specified sparsification 
techniques.
 
Users are encouraged to experiment with the training length and initial learning 
rate to either expedite training or to produce a more accurate model.

This can be done by either editing the recipe or supplying the `--recipe_args` 
argument to the training commands.

For example, the following appended to the training commands will change the number 
of epochs and the initial learning rate:

```bash
--recipe_args '{"num_epochs":8,"init_lr":0.0001}'
```

The following template can be used to apply this recipe in a `one-shot` manner:

```python
from sparseml.pytorch.optim import ScheduledModifierManager

model = ... # instantiated model
recipe = ... # local path to this recipe

manager = ScheduledModifierManager.from_yaml(recipe)
manager.apply(model) 
```
"""
