# PruneGPT: Exploring the effects of model compression GPT-*
Author: @spacemanidol

While there has been a lot of work exploring how models like BERT are effected by pruning there has been little work on pruning generative models or models that leverage transformer decoders such as GPT-2. To evaluate how pruning effects language modeling we fine tuned GPT-2 on the wikitext collection of datasets and evaluated perplexity on the validation portion of the dataset. In order to reproduce the runs please run the following commands with the desired modification to the pruning recipe and target dataset. Training will take 3 hours for wikitext 2 and about 1 week for wikitext-103 using a single 2080ti using a batch size of 1.  
```sh
python src/run_language_modeling.py --model_name_or_path gpt2 --do_eval --do_train --output_dir baseline-wikitext-2 --evaluation_strategy epoch --overwrite_output_dir --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --cache_dir cache/ --save_strategy epoch --seed 42 --recipe recipes/noprune.yaml --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --num_train_epochs 10
```
### Results
#### Wikitext-2
| Model                     | Sparsity |Perplexity|Distilled |
|---------------------------|----------|----------|----------|
| GPT-2 (XL)(Zero-Shot)1.5 B|0         |18.34     |No        |
| GPT-2 122M (Zero-Shot)    |0         |30.63     |No        |
| GPT-2 122M                |0         |21.76     |No        |
| GPT-2 122M                |0         |          |Yes       |
| GPT-2 122M                |90        |156.08    |No        |
| GPT-2 122M                |90        |          |Yes       |

#### Wikitext-103
| Model                     | Sparsity |Perplexity|Distilled |
|---------------------------|----------|----------|----------|
| GPT-2 (XL)(Zero-Shot)1.5 B|0         |17.48     |No        |
| GPT-2 122M (Zero-Shot)    |0         |30.63     |No        |
| GPT-2 122M                |0         |15.08     |No        |
| GPT-2 122M                |0         |          |Yes       |
| GPT-2 122M                |90        |26.79     |No        |
| GPT-2 122M                |90        |          |Yes       |


`