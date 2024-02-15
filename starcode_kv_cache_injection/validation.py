import onnxruntime as ort
import numpy as np
import onnx
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from onnx.tools import update_model_dims
from sparseml.onnx.utils import ONNXGraph
import logging
import numpy
from typing import List, Union

_LOGGER = logging.getLogger(__name__)


def create_causal_mask(
    input_ids: Union[numpy.ndarray, List[int]],
    attention_mask: Union[numpy.ndarray, List[int]],
    dtype: numpy.dtype = numpy.int64,
) -> numpy.ndarray:
    """
    Compute a causal mask from a set of module inputs.
    In transformers, a causal mask is a boolean mask that is used to
    prevent information from future positions in a sequence from
    being used to predict the current position. Each element of the mask
    is set to 1 if the corresponding position in the input sequence
    is allowed to attend to positions up to and including that position,
    and 0 otherwise.

    in case of single-token input, the causal mask is an array
    of shape [1, 1, 1, sequence_length],
    (essentially the reshaped attention_mask)

    in case of a multi-token input, the causal mask is an array
    of shape [batch_size, 1, input_ids_length, sequence_length]
    it is a concatenation of a:
     - past (cache) causal mask
     - and a causal mask (a lower triangular matrix of 1's and 0's)
    e.g
    ```
    input_ids = [[1,2,3,4]]
    attention_mask = [[1,1,1,1,1,1]]

    causal_mask = [[[[ 1 1 | 1 0 0 0 ],
                     [ 1 1 | 1 1 0 0 ],
                     [ 1 1 | 1 1 1 0 ],
                     [ 1 1 | 1 1 1 1 ]]]]
    ```
    or
    ```
    input_ids = [[1,2,3,4]]
    attention_mask = [[0,0,1,1,1,1,1]]

    causal_mask = [[[[ 0 0 1 1 | 1 0 0 0 ],
                     [ 0 0 1 1 | 1 1 0 0 ],
                     [ 0 0 1 1 | 1 1 1 0 ],
                     [ 0 0 1 1 | 1 1 1 1 ]]]]
    ```

    :param input_ids: input ids of the model input
    :param attention_mask: attention mask of the model input
    :param dtype: data type of the mask
    :return: causal mask
    """
    if isinstance(input_ids, numpy.ndarray):
        batch_size, input_ids_length = input_ids.shape

    else:
        batch_size, input_ids_length = 1, len(input_ids)

    if isinstance(attention_mask, numpy.ndarray):
        sequence_length = attention_mask.shape[1]
    else:
        sequence_length = len(attention_mask)
        attention_mask = numpy.array(attention_mask)[None, ...]

    if input_ids_length == 1:
        causal_mask = numpy.reshape(attention_mask, (batch_size, 1, 1, sequence_length))
        return causal_mask.astype(dtype)

    causal_mask = numpy.tril(
        numpy.ones((batch_size, 1, input_ids_length, input_ids_length), dtype=dtype), 0
    )
    past_causal_mask = numpy.ones(
        (batch_size, 1, input_ids_length, sequence_length - input_ids_length),
        dtype=dtype,
    )
    causal_mask = numpy.concatenate((past_causal_mask, causal_mask), axis=-1)

    num_zeros = numpy.count_nonzero(attention_mask == 0)

    # changes to the original function
    causal_mask[:, :, num_zeros:, :] = 0
    causal_mask = causal_mask.reshape(1, sequence_length, 1, -1)

    return causal_mask

def apply_input_shapes(model, onnx_model_path, sequence_length, config):
    kv_cache_hidden_dim = config.n_embd // config.n_head
    cache_changes_in = {n.name: [1, "dynamic_len_1", 2 * kv_cache_hidden_dim] for n in model.graph.input if n.name.startswith("past_key_values")}
    cache_changes_out = {n.name: [1, "dynamic_len_2", 2 * kv_cache_hidden_dim] for n in model.graph.output if n.name.startswith("present")}
    graph = ONNXGraph(model)

    graph.delete_unused_initializers()
    graph.delete_orphaned_node_branches()
    graph.sort_nodes_topologically()

    model = update_model_dims.update_inputs_outputs_dims(model,
                                                         {"input_ids": [1, "dynamic_len_3"],
                                                          "positions": [1, "dynamic_len_4"],
                                                          "attention_mask": [1, sequence_length],
                                                          "causal_mask": [1, "dynamic_len_5", 1, "dynamic_len_6"],
                                                          **cache_changes_in},

                                                          {"logits": [1, "dynamic_len_6", config.vocab_size], **cache_changes_out})

    onnx.save(model, onnx_model_path)
    return model


def multitoken_inference_test(onnx_model_path, prompt, config, tokenizer, sequence_length, logits_gt, kv_cache_gt):
    # feed the whole sequence to the model so that we can initially validate
    # the correctness of the kv cache injected model
    kv_cache_hidden_dim = config.n_embd // config.n_head
    inputs = tokenizer(prompt, return_tensors="np", padding='max_length', max_length=sequence_length)
    input_ids = inputs.input_ids  # (1, sequence_length)
    attention_mask = inputs.attention_mask  # (1, sequence_length)
    kv_cache = {f"past_key_values.{i}": np.zeros((1, 0, 2 * kv_cache_hidden_dim), dtype=np.float32) for i in
                range(config.n_layer)}  # (1, 0, 2 * embedding [because we have k and v's concatenated])
    causal_mask = create_causal_mask(input_ids, attention_mask)  # (1, sequence_length, 1, sequence_length)
    positions = attention_mask.cumsum(-1) - 1  # (1, sequence_length)

    session = ort.InferenceSession(onnx_model_path)

    out = session.run(
        None,
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            **kv_cache,
            "causal_mask": causal_mask,
            "positions": positions,
        },
    )
    logits, *kv_cache = out

    num_tokens_processed = logits_gt.shape[1] # only test the relevant, non-padded tokens
    assert np.allclose(logits[:, :num_tokens_processed, :], logits_gt, atol=1e-3)
    assert all(np.allclose(x[:, :num_tokens_processed, :], y, atol=1e-3) for x, y in zip(kv_cache, kv_cache_gt))

def singletoken_inference_test(onnx_model_path, prompt, config, tokenizer, sequence_length, logits_gt, kv_cache_gt):
    # feed the model one token at a time to validate the correctness of the kv cache injected model
    model = onnx.load(onnx_model_path, load_external_data=True)
    apply_input_shapes(model, onnx_model_path, sequence_length, config)

    kv_cache_hidden_dim = config.n_embd // config.n_head
    inputs = tokenizer(prompt, return_tensors="np")
    attention_mask = np.zeros((1, sequence_length), dtype=np.int64)
    kv_cache = {f"past_key_values.{i}": np.zeros((1,sequence_length-1, 2 * kv_cache_hidden_dim), dtype=np.float32) for i in range(config.n_layer)}
    session = ort.InferenceSession(onnx_model_path)

    for idx, token in enumerate(inputs.input_ids[0]):
        if token == tokenizer.pad_token_id:
            break
        attention_mask[:, -(idx + 1):] = 1
        positions = np.array([[idx]])
        input_ids = np.array([[token]])
        causal_mask = create_causal_mask(input_ids, attention_mask)

        outputs = session .run(None, {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "positions": positions,
            "causal_mask": causal_mask,
            **kv_cache
        })
        # will not run without throwing an error, there are some missing pieces that need to be addressed

def get_baseline(prompt, hf_model_name, tokenizer):
    model = AutoModelForCausalLM.from_pretrained(hf_model_name)
    tokens = tokenizer.encode(prompt, return_tensors="pt")
    out = model(tokens, return_dict=True)
    logits_gt = out.logits.detach().numpy()
    kv_cache_gt = [t.detach().numpy() for t in out.past_key_values]
    return logits_gt, kv_cache_gt

def main(prompt, hf_model_name, onnx_model_path, sequence_length):
    config = AutoConfig.from_pretrained(hf_model_name)
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    logits_gt, kv_cache_gt = get_baseline(prompt, hf_model_name, tokenizer)

    multitoken_inference_test(onnx_model_path, prompt, config, tokenizer, sequence_length, logits_gt, kv_cache_gt)
    _LOGGER.info("Successfully ran multi-token inference on the kv cache injected model")
    singletoken_inference_test(onnx_model_path, prompt, config, tokenizer, sequence_length, logits_gt, kv_cache_gt)
    _LOGGER.info("Successfully ran single-token inference on the kv cache injected model")



if __name__ == "__main__":
    PROMPT = "def eight_queens():\n    if True:\n        return 1\n    "
    HF_MODEL_NAME = "bigcode/tiny_starcoder_py"
    ONNX_MODEL_PATH = "/Users/damian/Code/nm/sparseml/tiny_starcoder_py/deployment/test.onnx"
    SEQUENCE_LENGTH = 256
    main(PROMPT, HF_MODEL_NAME, ONNX_MODEL_PATH, SEQUENCE_LENGTH)






