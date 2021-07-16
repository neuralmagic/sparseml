# PruneGPT: Exploring the effects of model compression GPT-*
Author: @spacemanidol


### Results
python src/run_language_modeling.py --model_name_or_path gpt2 --do_eval --output_dir base --evaluation_strategy epoch --overwrite_output_dir --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --cache_dir cache/ --save_strategy epoch --seed 42 --recipe recipes/90sparse.yaml --do_train --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1
#### Linguistic Reasoning
| Model           | Sparsity | Wikitext-103 PPL | CNN/NewsGroup|
|-----------------|----------|------------------|--------------|
| GPT-2 (XL)1.5 B |0         |17.48             |              |
| GPT-2 (XL)1.5 B |0         |17.48             |              |
| GPT-2 
-