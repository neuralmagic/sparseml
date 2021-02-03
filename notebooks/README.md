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


Assuming you are running the notebooks from within a virtual environment (recommended), you may follow the steps
below to prepare and launch your notebooks:

1. Create a kernel attached to the environment:

```bash
python -m ipykernel install --user --name your_env --display-name "Python (your_env)".
```

This kernel should then be available for you under the "Kernel > Change kernel" menu item.

2. If a notebook displays TensorBoard and you are running it from a remote server, you may forward the
port that TensorBoard uses (by default 6006) to your local machine:

```bash
ssh -N -f -L localhost:6006:localhost:6006 user@remote_ip_address
```

**Tip:** If the port is unavailable, you may look for the process using it with `sudo lsof -i :6006` and release it with
`kill -9 <PROCESS_ID>`. The above binding command also allows you to view TensorBoard outside your notebook by going to
`localhost:6006` from your local machine.

3. Some notebooks may make use of the [ipywidgets](https://github.com/jupyter-widgets/ipywidgets) package.
You may need to enable the Jupyter extension to properly see the UIs with the following command:

```bash
jupyter nbextension enable --py widgetsnbextension.
```

4. Start a Jupyter session in the `notebooks` directory, optionally using an available port of your choice (e.g., 8890):

```bash
cd notebooks
jupyter notebook --port=8890
```

Again, if you are running the Jupyter server from a remote server, you may bind the notebook port as you did with TensorBoard, then
view it from your local machine with `localhost:8890`.


Note, the TensorFlow V1 notebooks are tested with TensorFlow version ~= 1.15.0. 
For best results, confirm your system matches that version.

| Script     |      Description      |
|----------|-------------|
| [Keras Classification](https://github.com/neuralmagic/sparseml/blob/main/notebooks/keras_classification.ipynb)  | Notebook demonstrating how to prune a Keras classification model using SparseML |
| [PyTorch Classification](https://github.com/neuralmagic/sparseml/blob/main/notebooks/pytorch_classification.ipynb)  | Notebook demonstrating how to prune a PyTorch classification model using SparseML |
| [PyTorch Detection](https://github.com/neuralmagic/sparseml/blob/main/notebooks/pytorch_detection.ipynb)  | Notebook demonstrating how to prune a PyTorch detection model using SparseML |
| [TensorFlow V1 Classification](https://github.com/neuralmagic/sparseml/blob/main/notebooks/tensorflow_v1_classification.ipynb)  | Notebook demonstrating how to prune a TensorFlow V1 classification model using SparseML |
