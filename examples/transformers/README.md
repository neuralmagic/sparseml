# BERT-PRUNE-EXP
Experiments in pruning BERT

## Training
```python
python run_qa.py  \
 --model_name_or_path bert-base-uncased \
 --dataset_name squad \
 --do_train \
 --do_eval \
 --per_device_train_batch_size 12 \
 --per_device_eval_batch_size 12 \
 --learning_rate 3e-5 \
 --num_train_epochs 10 \
 --max_seq_length 384 \
 --doc_stride 128 \
 --output_dir output/ \
 --overwrite_output_dir \
 --cache_dir cache \
 --preprocessing_num_workers 4 \
 --seed 42 \
 --nm_prune_config prune-config.yaml 
```

## Model Performance

| model name        	| sparsity 	| total train epochs 	| prune epochs 	| F1 Score 	| EM Score  	|
|-------------------	|----------	|--------------------	|--------------	|----------	|-----------	|
| bert-base-uncased 	|0        	|2                  	|0            	|88.002     |80.634         |
| bert-base-uncased 	|0        	|10                 	|0            	|87.603     |79.130         |
| bert-base-uncased 	|80       	|10                 	|8            	|6.0676    	|0.312        	|
| bert-base-uncased 	|80       	|10                  	|8          	|83.951     |74.409         |
| bert-base-uncased 	|95       	|10                  	|8           	|87.603    	|79.130         |
| bert-base-uncased 	|95       	|10                 	|1            	|22.626   	|13.538         |
|                   	|          	|                    	|              	|          	|           	|
| bert-base-uncased 	| 9        	|                    	|              	|          	|           	|
| bert-base-uncased 	| 99        | 10                   	|  10           |87.603     |79.13          |
| bert-base-uncased 	| 9        	|                    	|              	|          	|           	|

