# Transformers and Sparseml
This folder contains examples on how to use sparseml with transformers. To show how to use the two libraries we have modified the SQUAD example from transformers and trained various models with varying sparsities/train schedules. The performance of the pretrained models can be found below.

## Training
To custom prune a model first go to the prune-config.yaml file and modify the parameters to your needs. !EpochRangeModifier controls how long the model trains for. Each !GMPruningModifier modifies controls how each portion is pruned. You can modify end_epoch to control how long the pruning regime lasts and final_sparsity and init_sparsity define the speed which the module is pruned and the final sparsity. 

Once you have updated prune-config.yaml updated to your pruning goals go ahead and run the command below. This will produce a trained model with an optimized onnx model. 
```python
python run_qa.py  \
 --model_name_or_path bert-base-uncased \
 --dataset_name squad \
 --do_train \
 --do_eval \
 --per_device_train_batch_size 12 \
 --per_device_eval_batch_size 12 \
 --learning_rate 3e-5 \
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
| bert-base-uncased 	|80       	|2                   	|1            	|6.0676    	|0.312        	|
| bert-base-uncased 	|80       	|10                  	|8          	|83.951     |74.409         |
| bert-base-uncased 	|95       	|10                  	|8           	|87.603    	|79.130         |
| bert-base-uncased 	|95       	|10                 	|0            	|87.603  	|79.130         |
| bert-base-uncased 	|99         |10                   	|10             |87.603     |79.130         |
| bert-base-uncased 	|99         |10                    	|0             	|87.603    	|79.130       	|

