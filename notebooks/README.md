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

## Tutorials for SparseML
Tutorials, which are implemented as Jupyter Notebooks for easy consumption and editing, 
are provided under the `notebooks` directory.
To run one of the tutorials, start a Jupyter session in the `notebooks` directory.
```bash
cd notebooks
jupyter notebook
```

Additionally, some notebooks may make use of the [ipywidgets](https://github.com/jupyter-widgets/ipywidgets) package.
You may need to enable the Jupyter extension to properly see the UIs.
Use the following command to do so: `jupyter nbextension enable --py widgetsnbextension`.
If Jupyter was already running, restart after running the command.

Once the Jupyter session has started, you can open the desired notebooks.
Note, the TensorFlow V1 notebooks are tested with TensorFlow version ~= 1.15.0. 
For best results, confirm your system matches that version.

| Script     |      Description      |
|----------|-------------|
| [Keras Classification](https://github.com/neuralmagic/sparseml/blob/main/notebooks/keras_classification.ipynb)  | Notebook demonstrating how to prune a Keras classification model using SparseML |
| [PyTorch Classification](https://github.com/neuralmagic/sparseml/blob/main/notebooks/pytorch_classification.ipynb)  | Notebook demonstrating how to prune a PyTorch classification model using SparseML |
| [PyTorch Detection](https://github.com/neuralmagic/sparseml/blob/main/notebooks/pytorch_detection.ipynb)  | Notebook demonstrating how to prune a PyTorch detection model using SparseML |
| [TensorFlow V1 Classification](https://github.com/neuralmagic/sparseml/blob/main/notebooks/tensorflow_v1_classification.ipynb)  | Notebook demonstrating how to prune a TensorFlow V1 classification model using SparseML |
