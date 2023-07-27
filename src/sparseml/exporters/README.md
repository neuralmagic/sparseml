# Exporters

## KV Cache Injector

The KV Cache Injector is a SparseML exporter that transforms the (sparsified)  ONNX Large Language Model (LLM) to support the key-value cache (KV cache) optimization.

Short description of the KV cache optimization and its benefits:

> In the context of transformers, key-value cache is a cache mechanism used to speed up the self-attention calculation in the transformer model.
> During the self-attention calculation, every token in the input sequence is compared to every other token to determine the importance of each token in the context of the others.
> This involves computing a dot product between the query vector of each token and the key vectors of all the other tokens. Analogous operation happens for value vectors as well. This operation can be computationally expensive, especially for long sequences.
>
> At the time of inference, we may speed up this process, the transformer model uses a cache of key-value pairs. The keys and values correspond to the output of the previous self-attention operation.
>When computing the self-attention for a new sequence, the model can look up the values corresponding to the keys in the cache, rather than recomputing them from scratch.
>This can significantly reduce the computational cost of the self-attention operation.

### Usage
Below you wil find an example of how to use the KV Cache Injector to transform an ONNX LLM model to support the KV cache optimization.

#### Model export
Assuming that the (sparsified) torch model is located at `torch_model_path`, you can export the model to ONNX using the following command:
```bash
sparseml.transformers.export_onnx --model_path torch_model_path --sequence_length 128 --task text-generation
```
this will save the exported model to a `model_path` directory.

Note: The KV cache injection is currently only supported for the models that are exported using one of the following libraries:
- [Neural Magic Transformers fork](https://github.com/neuralmagic/transformers) 
- [HuggingFace Transformers Version 4.30.2](https://pypi.org/project/transformers/4.30.2/)

We can not guarantee that the KV cache injection will work for the models exported using other versions of the HuggingFace Transformers library.

#### KV cache injection
Once the model has been exported to ONNX, you can use the KV Cache Injector to transform the model to support the KV cache optimization.

```python
import os
import onnx
from sparseml.exporters.kv_cache_injector import KeyValueCacheInjector

model_path = "..."

# Load the model
# Note: load_external_data=False is required to avoid loading the external data
# (i.e. the model's weights) into memory (could be a significant overhead for large models)
model = onnx.load(os.path.join(model_path, "model.onnx"), load_external_data=False)
# Apply the KV Cache Injector to the model graph
model = KeyValueCacheInjector(model_path).apply(model) 
# Overwrite the model with the version that has the KV cache "injected"
onnx.save(model, os.path.join(model_path, "model.onnx"))
```
Note: We will be adding support for the new LLM models to the `KeyValueCacheInjector`.
To check which LLM models are currently supported, run:

```python
from sparseml.exporters.kv_cache_injector import KeyValueCacheInjector

print(KeyValueCacheInjector.supported_models)
```

#### Run the model with the KV cache support in DeepSparse Engine
The KV cache support allows to speed up the inference of the LLM models in DeepSparse Engine. 

Once the model has been exported to ONNX and the KV cache has been injected, you can run the model with the KV cache support in DeepSparse Engine using the following command:

```python
from deepsparse import Pipeline

model_path = ...

tg_pipeline = Pipeline.create(
    task="text-generation",
    model_path=model_path,
)
out = tg_pipeline(sequences="Who is the president of the United States?")
print(out)
```

For more information on how to run LLMs in DeepSparse Engine, please refer to the [DeepSparse Transformers documentation](...)


