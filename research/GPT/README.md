# PruneGPT: Exploring the effects of model compression GPT-*
Author: @spacemanidol

While there has been a lot of work exploring how models like BERT are effected by pruning there has been little work on pruning generative models or models that leverage transformer decoders such as GPT-2. To evaluate how pruning effects language modeling we fine tuned GPT-2 on the wikitext collection of datasets and evaluated perplexity on the validation portion of the dataset. In order to reproduce the runs please run the following commands with the desired modification to the pruning recipe and target dataset. Training will take 3 hours for wikitext 2 and about 1 week for wikitext-103 using a single 2080ti using a batch size of 1. 

```sh
python src/run_language_modeling.py --model_name_or_path gpt2 --do_eval --do_train --output_dir baseline-wikitext-2 --evaluation_strategy epoch --overwrite_output_dir --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --cache_dir cache/ --save_strategy epoch --seed 42 --recipe recipes/noprune.yaml --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --num_train_epochs 10
```
### Results
#### Wikitext-2
| Model                     | Sparsity |Perplexity|Distilled |Teacher          |
|---------------------------|----------|----------|----------|-----------------|
| GPT-2 (XL)(Zero-Shot)1.5 B|0         |18.34     |No        |N/A              |
| GPT-2 122M (Zero-Shot)    |0         |30.63     |No        |N/A              |
| GPT-2 122M                |0         |21.76     |No        |N/A              |
| GPT-2 122M                |0         |1.33e9    |Yes       |gpt2(zero-shot)  |
| GPT-2 122M                |0         |1.43e13   |Yes       |gpt2(wikitext2)  |
| GPT-2 122M                |0         |2.76e 70  |Yes       |gpt2(wikitext103)|
| GPT-2 122M                |90        |156.08    |No        |N/A              |
| GPT-2 122M                |90        |inf       |Yes       |gpt2(zero-shot)  |
| GPT-2 122M                |90        |inf       |Yes       |gpt2(wikitext2)  |
| GPT-2 122M                |90        |inf       |Yes       |gpt2(wikitext103)|

#### Wikitext-103
| Model                     | Sparsity |Perplexity|Distilled |Teacher          |
|---------------------------|----------|----------|----------|-----------------|
| GPT-2 (XL)(Zero-Shot)1.5 B|0         |17.48     |No        |N/A              |
| GPT-2 122M (Zero-Shot)    |0         |30.63     |No        |N/A              |
| GPT-2 122M                |0         |15.08     |No        |N/A              |
| GPT-2 122M                |0         |          |Yes       |gpt2(zero-shot)  |
| GPT-2 122M                |0         |          |Yes       |gpt2(wikitext2)  |
| GPT-2 122M                |0         |          |Yes       |gpt2(wikitext103)|
| GPT-2 122M                |90        |26.79     |No        |N/A              |
| GPT-2 122M                |90        |          |Yes       |gpt2(zero-shot)  |
| GPT-2 122M                |90        |          |Yes       |gpt2(wikitext2)  |
| GPT-2 122M                |90        |          |Yes       |gpt2(wikitext103)|


##### Individual Layer Pruning on Wikitext-2

To explore sensitivity of various layers to pruning we alterante pruning individual layers of the model to 80% sparsity and evalute the impact on overall perplexity. Overall, we do not see a major difference in the perplexity of the model with variations in which layers are pruned but we do note a minor increase in perplexity when early or later layers are pruned.

| Model                     | Layer Pruned |Perplexity|
|---------------------------|--------------|----------|
| GPT-2 (XL)(Zero-Shot)1.5 B|N/A           |18.34     |
| GPT-2 122M (Zero-Shot)    |N/A           |30.63     |
| GPT-2 122M                |N/A           |21.76     |
| GPT-2 122M                |0             |23.62     |
| GPT-2 122M                |1             |22.94     |
| GPT-2 122M                |2             |23.06     |
| GPT-2 122M                |3             |23.06     |
| GPT-2 122M                |4             |23.27     |
| GPT-2 122M                |5             |23.41     |
| GPT-2 122M                |6             |23.48     |
| GPT-2 122M                |7             |23.41     |
| GPT-2 122M                |8             |23.41     |
| GPT-2 122M                |9             |23.48     |
| GPT-2 122M                |10            |23.57     |
| GPT-2 122M                |11            |23.74     |
