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

import torch

from sparseml.experimental.sparsegpt.dispatch import evaluate_perplexity, load_model
from sparseml.experimental.sparsegpt.main import sequential
from sparseml.experimental.sparsegpt.opt import load_data
from sparseml.modifiers.obcq.utils.helpers import ppl_eval_general
from sparseml.transformers.sparsification.obcq.obcq import one_shot
from sparseml.transformers.sparsification.obcq.utils.helpers import opt_forward


dataset = "c4"
model_name = "facebook/opt-1.3b"
sparsity = 0.5
nbits = 8
smooth_quant = 0
observer_batches = 128
nsamples = 128
data_sequence_length = 2048
sequential_hessian = 0
experimental_recipe = "src/sparseml/experimental/sparsegpt/examples/opt/recipes/"
experimental_recipe += "opt-1.3b-opt_pretrain-pruned50_quantW8A8.md"
prod_recipe = "src/sparseml/transformers/sparsification/obcq/example.yaml"
device = "cuda:0"
seed = 0
prunen = 0
prunem = 0
percdamp = 0.01
blocksize = 128
ptq_only = 0


class ExperimentalArgs:
    model = model_name
    dataset = dataset
    data_sequence_length = data_sequence_length
    sequential_hessian_within_layer = sequential_hessian
    recipe = experimental_recipe
    sparsity = sparsity
    wbits = nbits
    observer_batches = observer_batches
    nsamples = nsamples
    smoothquant = smooth_quant
    seed = seed
    prunen = prunen
    prunem = prunem
    percdamp = percdamp
    blocksize = blocksize
    ptq_only = ptq_only


class ProdArgs:
    model = model_name
    dataset = dataset
    nsamples = nsamples
    device = device
    recipe = prod_recipe
    save = False


def run_experimental_obcq(experimental_args):
    model, _ = load_model(experimental_args)
    calibration_data, _, _ = load_data(experimental_args, data_sequence_length)
    sequential(model, calibration_data, device, experimental_args)

    del calibration_data
    return model


if __name__ == "__main__":
    experimental_args = ExperimentalArgs()
    exp_model = run_experimental_obcq(experimental_args)
    experimental_args.dataset = "wikitext2"
    _, testloader, _ = load_data(experimental_args, data_sequence_length)
    exp_perplexity = evaluate_perplexity(
        experimental_args, exp_model, testloader, device, max_samples_per_iteration=8
    )

    del testloader
    del exp_model
    torch.cuda.empty_cache()

    prod_args = ProdArgs()
    prod_model = one_shot(
        model_path=prod_args.model,
        dataset_name=prod_args.dataset,
        num_samples=prod_args.nsamples,
        device=prod_args.device,
        recipe_file=prod_args.recipe,
    )
    experimental_args.dataset = "wikitext2"
    _, testloader, _ = load_data(experimental_args, data_sequence_length)
    prod_perplexity = ppl_eval_general(
        opt_forward, prod_model, testloader, device, max_samples_per_iteration=8
    )
    print(
        f"Experimental Perplexity: {exp_perplexity}, "
        f"Production Perplexity: {prod_perplexity}"
    )
