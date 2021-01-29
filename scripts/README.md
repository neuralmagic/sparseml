<!--
Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Scripts for SparseML

This directory provides easy-of-use scripts for performing common SparseML work
flows.  This includes
model training, pruning, quantization, exporting, and sensitivity analysis.

These scripts natively support models in the SparseML submodules, however can be adapted for use with your own models
or used to inspire new workflows.  To see examples of simple integrations with SparseML check out our
[notebooks](https://github.com/neuralmagic/sparseml/tree/main/notebooks)
and [examples](https://github.com/neuralmagic/sparseml/tree/main/examples).  

To run one of the scripts, invoke it with a Python command from the command line along with the relevant arguments.

```bash
python scripts/<SCRIPT>.py <TASK-NAME> <KW-ARGUMENTS>
```

Each script file is fully documented with descriptions, command help printouts, and example commands.  You can also
run any script with `-h` or `--help` to see the help printout.

## Scripts

| Script     |      Description      |
|----------|-------------|
| [PyTorch Vision](https://github.com/neuralmagic/sparseml/blob/main/scripts/pytorch_vision.py)  | Script for training, optimization, export, pruning sensitivity analysis, or learning rate sensitivity analysis of PyTorch classification and Detection models |
| [TensorFlow V1 Classification](https://github.com/neuralmagic/sparseml/blob/main/scripts/tensorflow_v1_classification.py)  | Script for training, optimization, export, or pruning sensitivity analysis of TensorFlow V1 classification models  |