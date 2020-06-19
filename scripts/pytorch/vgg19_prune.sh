CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/pytorch/classification_recal.py \
	--recal-config-path scripts/pytorch/configs/pruning_vgg19.yaml \
	--arch-key vgg19 --dataset imagenet --dataset-path /dev/shm/ILSVRC \
	--train-batch-size 256 --test-batch-size 2048 --loader-num-workers 24 \
	--save-epochs 79 85 87 

