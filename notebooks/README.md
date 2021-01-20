## Tutorials for SparseML
Tutorials, which are implemented as Jupyter Notebooks for easy consumption and editing, 
are provided under the `notebooks` directory. 
To run one of the tutorials, start a Jupyter session in the `notebooks` directory.
```bash
cd notebooks
jupyter notebook
```

Additionally, the notebooks make use of the [ipywidgets](https://github.com/jupyter-widgets/ipywidgets) package.
You may need to enable the Jupyter extension to properly see the UIs.
Use the following command to do so: `jupyter nbextension enable --py widgetsnbextension`.
If Jupyter was already running, restart after running the command.

Once the Jupyter session has started, you can open the desired notebooks.
Note, the notebooks are tested with TensorFlow version ~= 1.15.0. 
For best results, make sure your system matches that version.

### model_repo.ipynb
A tutorial for exploring and downloading from the [SparseZoo](https://docs.neuralmagic.com/sparsezoo/). 
It allows downloads for any model currently available within any ML framework.

### pruning_adam_pytorch.ipynb
A tutorial for pruning models in PyTorch using an Adam optimizer. 
A step-by-step process, along with simple UIs, is given to make the process easier and more intuitive. 
It is used to increase the performance of models when executing in the Neural Magic Inference Engine.

### pruning_adam_tensorflow.ipynb
A tutorial for pruning models in TensorFlow using an Adam optimizer. 
A step-by-step process, along with simple UIs, is given to make the process easier and more intuitive. 
It is used to increase the performance of models when executing in the Neural Magic Inference Engine.

### quantize_model_post_training.ipynb
A tutorial for using `sparseml.onnx.quantization` to perform post-training quantization on a
trained model from an ONNX file. A step-by-step example is given that walks through preparing a
calibration dataset, quantizing the model, and validating the results.

### transfer_learning_pytorch.ipynb
A tutorial for transfer learning from a model in the [SparseZoo](https://docs.neuralmagic.com/sparsezoo/)
within PyTorch using an Adam optimizer.
The main use case is transfer learning from a previously pruned model. 
In this way, you can limit the training time needed as well as the potential complexity of 
the pruning process while keeping the performance.

### transfer_learning_tensorflow.ipynb
A tutorial for transfer learning from a model in the [SparseZoo](https://docs.neuralmagic.com/sparsezoo/)
within TensorFlow using an Adam optimizer.
The main use case is transfer learning from a previously pruned model.
In this way, you can limit the training time needed as well as the potential complexity of
the pruning process while keeping the performance.