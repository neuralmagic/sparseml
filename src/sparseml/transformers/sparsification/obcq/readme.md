# One Shot With SparseML 
This page walks through how to perform one-shot quantization of large language models using [SparseML](https://github.com/neuralmagic/sparseml). 

## Table of Contents 
1. [How to Clone and Install  the Latest SparseML](#clone)
2. [How to One-shot TinyLlama](#TinyLlama)
3. [How to Evaluate the One-shot Model](#evaluate)
4. [How to Export the One-shot model](#export)
5. [How to Inject KV Cache](#kvcache)
6. [Using the Model With DeepSparse](#DeepSparse)
7. [Upload Model to Hugging Face](#upload)
8. [Explaining the TinyLlama Recipe](#recipe)
9. [How to Adapt a Recipe for a New Model](#adapt)


## <a name="clone">How to Clone and Install  the Latest SparseML </a>
You'll need the latest version of SparseML to run the one-shot workflow. To avoid any issues, we recommend that you do this in a fresh Python environment. 

Clone the SparseML repo: 
```bash
git clone https://github.com/neuralmagic/sparseml
```

Install the required dependencies: 
```bash
pip install -e "sparseml[transformers]" "torch==2.1"
```

## <a name="TinyLlama">How to One-shot TinyLlama </a>
[TinyLlama-1.1B-Chat](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v0.4) is an LLM that we can quantize in a short amount of time becasue it has 1.1B parameters. The steps are as follows: 

Change into the SparseML directory:

```bash
cd sparseml
```
Perform one-shot using the OBCQ algorithm. The command takes the following parameters: 
-  `model_path` a path to Hugging Face stub
- `dataset_name` Hugging Face dataset to extract calibration data from
- `num_samples` number of samples to extract from the dataset, defaults to 512
- `deploy_dir` the directory where the model will be saved, defaults to `obcq_deployment`
- `recipe_file` the file containing the one-shot hyperparameters 
- `precision` precision to load model as, either auto (default), half or full
- `eval_data` dataset to use for perplexity evalaution, or none to skip
- `do_save` whether to save the output model to disk
```bash
python sparsemlsrc/sparseml/transformers/sparsification/obcq/obcq.py TinyLlama/TinyLlama-1.1B-Chat-v0.4 open_platypus --recipe recipe.yaml --save True
```
## <a name="evaluate"> How to Evaluate the One-shot Model</a>
Next, evaluate the perforamnce of the model using the [lm-evaluation-harness framework](https://github.com/neuralmagic/lm-evaluation-harness).

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
%%bash 
cd lm-evaluation-harness
git checkout sparse_new_modifier
PRETRAINED_PATH=/home/mwitiderrick/neuralmagic/sparseml/obcq_deployment
TASK=hellaswag
BATCH_SIZE=64
SHOTS=0
DEVICE="cuda:0"

start=`date +%s`
python main_sparse.py \
 --model hf-causal-experimental \
 --model_args pretrained=$PRETRAINED_PATH,trust_remote_code=True \
 --tasks $TASK \
 --batch_size $BATCH_SIZE \
 --no_cache \
 --write_out \
 --output_path "${PRETRAINED_PATH}/${TASK}.json" \
 --device "${DEVICE}" \
 --num_fewshot $SHOTS
 end=`date +%s`
 echo Execution time was `expr $end - $start` seconds.
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
Once you are certain the model is performing as expected, you can export it for inference. The `export.py` file provides the functions for doing this. Running the command below creates a `deployment` directory containing all the artifacts that are needed for running the model using DeepSparse. 

```bash
python sparseml/src/sparseml/transformers/sparsification/obcq/export.py --task text-generation --model_path obcq_deployment 

```

## <a name="kvcache">How to Inject KV Cache</a>
Injecting KV Cache is done to reduce the model’s computational overhead and speed up inference by caching the Key and Value states.
This is done by creating a copy of `model.onnx` and injecting the KV Cache:
```bash
cp deployment/model.onnx deployment/model-orig.onnx
python onnx_kv_inject.py --input-file deployment/model-orig.onnx --output-file deployment/model.onnx
```

## <a name="DeepSparse">Using the Model With DeepSparse </a>
Next, run inference using DeepSparse. Ensure you have the latest version of DeepSparse installed:

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
Check out the [DeepSparse pipeline text generation docs](https://github.com/neuralmagic/deepsparse/blob/main/src/deepsparse/transformers/text_generation.md) for full list of supported parameters. 

## <a name="uypload">Upload Model to Hugging Face</a>
You may want to upload the one-shot model to Hugging Face for ease of reference in the future or to share it with your colleagues. 

Start by installing `huggingface_hub`:
```
pip install huggingface_hub
```
Log into your Hugging Face account: 
```python
import huggingface_hub
huggingface_hub.login(token="HF_WRITE_TOKEN")
```
Head over to your [Hugging Face account](https://huggingface.co/new) and create a model named `TinyLlama-1.1B-Chat-v0.4-pruned50-quant`:
```python
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path="deployment",
    repo_id="YOUR_HF_USERNAME/TinyLlama-1.1B-Chat-v0.4-pruned50-quant",
    repo_type="model",
)
```

## <a name="recipe"> Explaining the TinyLlama Recipe</a>
The recipe below is what we used to one-shot the TinyLlama model. 
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
We will now walk through what the different hyperparameters mean and why they are set to those values.

## <a name="adapt"> How to Adapt a Recipe for a New Model</a>

The above recipe can be modified to perform one-shot on other models, for example Mistral. 
We can peform the following modifications on the recipe to one-shot a Mistral model.
- ff
- ff
Here is how the final recipe looks like: 
```yaml

```
