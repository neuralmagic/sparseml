# One Shot With SparseML 
This page describes how to perform one-shot quantization of large language models using [SparseML](https://github.com/neuralmagic/sparseml). This workflow requires a GPU with at least 16GB VRAM and 64GB of system RAM.

### Note on system requirements

Due to inefficiencies in PyTorch ONNX export, a lot of system memory is required to export the models for inference. There are [improvements coming in 2.2](https://github.com/pytorch/pytorch/commit/b4a49124c8165a374a3ef49e14807ac05b3fc030).

## Table of Contents 
1. [How to Clone and Install  the Latest SparseML](#clone)
2. [How to One-shot TinyLlama](#tinyllama)
3. [How to Evaluate the One-shot Model](#evaluate)
4. [How to Export the One-shot model](#export)
5. [How to Inject KV Cache](#kvcache)
6. [Using the Model With DeepSparse](#deepsparse)
7. [Upload Model to Hugging Face](#upload)
8. [Explaining the TinyLlama Recipe](#recipe)
9. [How to Adapt a Recipe for a New Model](#adapt)


## <a name="clone">How to Clone and Install  the Latest SparseML </a>
You'll need the latest version of SparseML to run the one-shot workflow. We recommend that you do this from source and in a fresh Python environment to avoid any issues. 

Clone the SparseML repo and install it locally: 
```bash
git clone https://github.com/neuralmagic/sparseml
pip install -e "sparseml[transformers]"
```

## <a name="tinyllama">How to One-shot TinyLlama </a>
[TinyLlama-1.1B-Chat](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v0.4) is an LLM that we can quantize in a short time because it has 1.1B parameters. 

Perform one-shot using the OBCQ algorithm. The command takes the following parameters: 

positional arguments:
- `model` a path to Hugging Face stub
- `dataset_name` Hugging Face dataset to extract calibration data from. Example of supported datasets: `{c4,evolcodealpaca,gsm8k,open_platypus,ptb,wikitext2}`

options:
- `--nsamples` number of samples to extract from the dataset, defaults to 512.
- `--deploy-dir` the directory where the model will be saved, defaults to `obcq_deployment`.
- `--eval` dataset to use for perplexity evaluation, or none to skip.
- `--save` whether to save the output model to disk.
- `--recipe` the file containing the one-shot hyperparameters.
- `--device` which device to load the model onto, either `cpu` or a specific `cuda:0`.
- `--precision` precision to load model as, either auto (default), half, full, float16 or float32.

Example command:
```bash
wget https://huggingface.co/nm-testing/TinyLlama-1.1B-Chat-v0.4-pruned50-quant/raw/main/recipe.yaml # download recipe
python sparseml/src/sparseml/transformers/sparsification/obcq/obcq.py TinyLlama/TinyLlama-1.1B-Chat-v0.4 open_platypus --recipe recipe.yaml --save True
```
## <a name="evaluate"> How to Evaluate the One-shot Model</a>
Next, evaluate the model's performance using the [lm-evaluation-harness framework](https://github.com/neuralmagic/lm-evaluation-harness).

Clone the repository:
```bash
git clone https://github.com/neuralmagic/lm-evaluation-harness.git
```
Install the required dependencies:
```bash
cd lm-evaluation-harness
pip install -e .
```
Evaluate on the `hellaswag` task:
```bash
git checkout sparse_new_modifier
start=`date +%s`
python main_sparse.py \
 --model hf-causal-experimental \
 --model_args pretrained=obcq_deployment,trust_remote_code=True \
 --tasks hellaswag \
 --batch_size 64 \
 --no_cache \
 --write_out \
 --output_path "obcq_deployment/hellaswag.json" \
 --device "cuda:0" \
 --num_fewshot 0
 end=`date +%s`
 echo Execution time was `expr $end - $start` seconds.
```
The results obtained in this case are:
```
Running loglikelihood requests
100%|██████████| 40145/40145 [20:47<00:00, 32.19it/s] 
{
  "results": {
    "hellaswag": {
      "acc": 0.40141406094403503,
      "acc_stderr": 0.004891826692722827,
      "acc_norm": 0.5115514837681737,
      "acc_norm_stderr": 0.004988449593007253
    }
  },
  "versions": {
    "hellaswag": 0
  },
  "config": {
    "model": "hf-causal-experimental",
    "model_args": {
      "pretrained": "/home/mwitiderrick/neuralmagic/sparseml/obcq_deployment",
      "trust_remote_code": true
    },
    "num_fewshot": 0,
    "batch_size": "64",
    "batch_sizes": [],
    "device": "cuda:0",
    "no_cache": true,
    "limit": null,
    "bootstrap_iters": 100000,
    "description_dict": {}
  }
}
hf-causal-experimental (pretrained=/home/mwitiderrick/neuralmagic/sparseml/obcq_deployment,trust_remote_code=True), limit: None, provide_description: False, num_fewshot: 0, batch_size: 64
|  Task   |Version| Metric |Value |   |Stderr|
|---------|------:|--------|-----:|---|-----:|
|hellaswag|      0|acc     |0.4014|±  |0.0049|
|         |       |acc_norm|0.5116|±  |0.0050|

Execution time was 1288 seconds.
```
Repeat the above on other tasks such as `truthfulqa-mc`, `winogrande`, and `drop`.
## <a name="export"> How to Export the One-shot Model</a>
Once you are certain the model is performing as expected, you can export it for inference. The `export.py` file provides the functions for doing this. Running the command below creates a `deployment` directory containing all the artifacts that are needed for inference with DeepSparse. 

```bash
python sparseml/src/sparseml/transformers/sparsification/obcq/export.py --task text-generation --model_path obcq_deployment 

```

## <a name="kvcache">How to Inject KV Cache</a>
Injecting KV Cache is done to reduce the model’s computational overhead and speed up inference by caching the Key and Value states.
This is done by creating a copy of `model.onnx` and injecting the KV Cache:
```bash
cp deployment/model.onnx deployment/model-orig.onnx
```

Code to inject KV Cache:
```python
import os
import onnx
from sparseml.exporters.kv_cache_injector import KeyValueCacheInjector
input_file = "deployment/model-orig.onnx"
output_file = "deployment/model.onnx"
model = onnx.load(input_file, load_external_data=False)
model = KeyValueCacheInjector(model_path=os.path.dirname(input_file)).apply(model)
onnx.save(model, output_file)
print(f"Modified model saved to: {output_file}")
```

## <a name="deepsparse">Using the Model With DeepSparse </a>
Next, run inference using DeepSparse. Ensure you have the latest version of DeepSparse installed with `pip install -U deepsparse-nightly[llm]`

```python
from deepsparse import TextGeneration

prompt = "How to get in a good university?"
formatted_prompt =  f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

model = TextGeneration(model="deployment")
print(model(formatted_prompt, max_new_tokens=200).generations[0].text)
"""
There are many factors to consider when choosing a university. Here are some tips for getting into a good university:

1. Research your options: Consider the schools in your area and the ones in your desired location. Research their reputation, tuition, and academic programs.

2. Apply to multiple universities: Apply to multiple universities, ensuring that you are applying to the best option for you.

3. Get a job: If you are applying to a university, you will need to find a job to support your studies. This will help you budget and manage your time.

4. Get involved with your community: Your university will likely have a community of students and faculty. Engage with this community by volunteering, participating in clubs, and engaging with others in your community.

5. Get involved with extracurricular activities: Universities often have many extracurricular activities, which can help you meet new people
"""
```
Check out the [DeepSparse pipeline text generation docs](https://github.com/neuralmagic/deepsparse/blob/main/src/deepsparse/transformers/text_generation.md) for the full list of supported parameters. 

## <a name="upload">Upload Model to Hugging Face</a>
You may want to upload the one-shot model to Hugging Face for ease of reference or to share it with your colleagues. 

Head over to your [Hugging Face account](https://huggingface.co/new) and create a model named `TinyLlama-1.1B-Chat-v0.4-pruned50-quant`. Then upload the one-shot model: 
```python
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path="deployment",
    repo_id="YOUR_HF_USERNAME/TinyLlama-1.1B-Chat-v0.4-pruned50-quant",
    repo_type="model",
    token="HF_WRITE_TOKEN"
)
```

## <a name="recipe"> Explaining the TinyLlama Recipe</a>
A recipe is a set of hyperparameters that provide detailed instructions on how the [one-shot quantization](https://neuralmagic.com/video/pruning-and-quantizing-ml-models-with-one-shot-without-retraining/) should be done. The recipe performs quantization in one shot, meaning that no retraining of the LLM is required. 

We will now walk through what the different hyperparameters mean and why they are set to those values.

The `SmoothQuantModifier` is a technique used for dealing with outliers in the weights and activations of the LLM because quantization is very sensitive to large variations in their values. For TinyLlama a `smoothing_strength` value of 0.8 resulted in a model with repetitions in its output but the problem was solved by lowering the value to 0.5. 

The `ignore` parameter under `QuantizationModifier` allows us to define operations that either don't make sense to quantize or operations that are too sensitive to quantize. Performing quantization on sensitive operations will affect the final accuracy of the model. We also don't quantize the inputs to the embedding layer. 

Under `SparseGPTModifier`, we define `sparsity` as 0.5 because we are aiming for a model that is 50% quantized. The other parameters are:
- `block_size` determines the number of columns to compress in one pass.
- `quantize` whether or not to quantize weights during SparseGPT.  A default quantization modifier will be applied when `quantize` is set to `True` and there is no `QuantizationModifier` in the recipe.
- `dampening_frac` amount of dampening to apply to H, as a fraction of the diagonal norm.
- `sequential_update` whether or not to update weights sequentially by layer, True saves on GPU memory.
- `mask_structure` string to define the structure of the mask to apply, "0:0" means that it's an unstructured mask. Setting it to "16:32" would mean that 16 out of every 32 weights will be zeroed out (structured sparsity).
- `targets` list of layer names to compress during OBCQ, or '__ALL__' to compress every layer in the model.

```yaml
test_stage:
  obcq_modifiers:
    SmoothQuantModifier:
      smoothing_strength: 0.5
      mappings: [
        [["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"], "re:.*input_layernorm"],
        [["re:.*gate_proj", "re:.*up_proj"], "re:.*post_attention_layernorm"]
      ]
    QuantizationModifier:
      ignore:
      # These operations don't make sense to quantize
      - LlamaRotaryEmbedding
      - LlamaRMSNorm
      - SiLUActivation
      # Skip quantizing the BMMs
      - QuantizableMatMul
      # Skip quantizing the layers with the most sensitive activations
      - model.layers.21.mlp.down_proj
      - model.layers.7.mlp.down_proj
      - model.layers.2.mlp.down_proj
      - model.layers.20.mlp.down_proj
      - model.layers.19.mlp.down_proj
      post_oneshot_calibration: true
      scheme_overrides:
        Embedding:
          input_activations: null
          weights:
            num_bits: 8
            symmetric: false
    SparseGPTModifier:
      sparsity: 0.5
      block_size: 128
      sequential_update: true
      quantize: true
      percdamp: 0.01
      mask_structure: "0:0"
      targets: ["re:model.layers.\\d*$"]
```
## <a name="adapt"> How to Adapt a Recipe for a New Model</a>
You can modify the above recipe to perform one-shot quantization on other models, for example [Mistral](https://huggingface.co/docs/transformers/main/model_doc/mistral). 

Perform the following modifications on the recipe to one-shot a Mistral model.
- Define the operations we want to skip during quantization, that is sensitive layers and operations that don't make sense to quantize.
- Declare the desired sparsity level, same as the one for TinyLlama.
- State the layers to compress during OBCQ.

Here is what the final recipe looks like: 
```yaml
test_stage:
  obcq_modifiers:
    SmoothQuantModifier:
      smoothing_strength: 0.5
      mappings: [
        [["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"], "re:.*input_layernorm"],
        [["re:.*gate_proj", "re:.*up_proj"], "re:.*post_attention_layernorm"]
      ]
    QuantizationModifier:
      ignore:
      # These operations don't make sense to quantize
      - MistralRotaryEmbedding
      - MistralRMSNorm
      - SiLUActivation
      # Skip quantizing the layers with the most sensitive activations
      - model.layers.1.mlp.down_proj
      - model.layers.31.mlp.down_proj
      - model.layers.30.mlp.down_proj
      - model.layers.30.mlp.gate_proj
      - model.layers.30.mlp.up_proj
      post_oneshot_calibration: true
      scheme_overrides:
        Embedding:
          input_activations: null
          weights:
            num_bits: 8
            symmetric: false
    SparseGPTModifier:
      sparsity: 0.5
      block_size: 128
      sequential_update: true
      quantize: true
      percdamp: 0.01
      mask_structure: "0:0"
      targets: ["re:model.layers.\\d*$"]
```

Save the recipe to a file named `recipe.yaml`. 

Run one-shot quantization on any Mistral-based model, for example, `zephyr-7b-beta`: 
```bash
python sparseml/src/sparseml/transformers/sparsification/obcq/obcq.py HuggingFaceH4/zephyr-7b-beta open_platypus --recipe recipe.yaml --precision float16 --save True
```
We set `precision` to `float16` because quantization is not supported for the `bfloat16` data type as of this writing. 

Repeat the other processes as shown previously. 

## Conclusion
In case of any questions, submit an [issue on GItHub](https://github.com/neuralmagic/sparseml) or join other LLM developers on our [community](https://join.slack.com/t/discuss-neuralmagic/shared_invite/zt-q1a1cnvo-YBoICSIw3L1dmQpjBeDurQ). 
