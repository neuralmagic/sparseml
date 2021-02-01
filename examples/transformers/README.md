# BERT-PRUNE-EXP
Experiments in pruning BERT

## Training
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

# Model Perf
bert-base-uncased 2 epochs training
2021-01-18 13:18:45 __main__     INFO     ***** Eval results *****
01/18/2021 13:18:45 - INFO - __main__ -   ***** Eval results *****
2021-01-18 13:18:45 __main__     INFO       exact_match = 80.63386944181646
01/18/2021 13:18:45 - INFO - __main__ -     exact_match = 80.63386944181646
2021-01-18 13:18:45 __main__     INFO       f1 = 88.00195611006883
01/18/2021 13:18:45 - INFO - __main__ -     f1 = 88.00195611006883
2021-01-18 13:18:45 __main__     INFO       epoch = 2.0
01/18/2021 13:18:45 - INFO - __main__ -     epoch = 2.0

bert-base-uncased 10 epochs training
021-01-18 18:30:15 __main__     INFO     ***** Eval results *****
01/18/2021 18:30:15 - INFO - __main__ -   ***** Eval results *****
2021-01-18 18:30:15 __main__     INFO       exact_match = 79.12961210974456
01/18/2021 18:30:15 - INFO - __main__ -     exact_match = 79.12961210974456
2021-01-18 18:30:15 __main__     INFO       f1 = 87.60263262023561
01/18/2021 18:30:15 - INFO - __main__ -     f1 = 87.60263262023561
2021-01-18 18:30:15 __main__     INFO       epoch = 10.0
01/18/2021 18:30:15 - INFO - __main__ -     epoch = 10.0

95% Sparsity 10 epochs(prune over first epoch)
2021-01-20 09:56:09 __main__     INFO     ***** Eval results *****
01/20/2021 09:56:09 - INFO - __main__ -   ***** Eval results *****
2021-01-20 09:56:09 __main__     INFO       exact_match = 13.538315988647115
01/20/2021 09:56:09 - INFO - __main__ -     exact_match = 13.538315988647115
2021-01-20 09:56:09 __main__     INFO       f1 = 22.62638896595549
01/20/2021 09:56:09 - INFO - __main__ -     f1 = 22.62638896595549

95% Sparsity 10 epochs(prune over first 8 epoch)
2021-01-20 15:09:58 __main__     INFO       exact_match = 79.12961210974456
01/20/2021 15:09:58 - INFO - __main__ -     exact_match = 79.12961210974456
2021-01-20 15:09:58 __main__     INFO       f1 = 87.60263262023561
01/20/2021 15:09:58 - INFO - __main__ -     f1 = 87.60263262023561


80% Sparsity 10 epochs(prune over first 8 epochs)
2021-01-18 11:54:30 __main__     INFO       exact_match = 74.40870387890256
01/18/2021 11:54:30 - INFO - __main__ -     exact_match = 74.40870387890256
2021-01-18 11:54:30 __main__     INFO       f1 = 83.95147251655385
01/18/2021 11:54:30 - INFO - __main__ -     f1 = 83.95147251655385

80% sparcity 2 epochs(prune in one epoch)
2021-01-18 12:08:08 __main__     INFO       exact_match = 0.3122043519394513
01/18/2021 12:08:08 - INFO - __main__ -     exact_match = 0.3122043519394513
2021-01-18 12:08:08 __main__     INFO       f1 = 6.067619413014725
01/18/2021 12:08:08 - INFO - __main__ -     f1 = 6.067619413014725


# Inference and Stuff
#ONNX path

1. Train model
2. Convert using convert
python convert_graph_to_onnx.py --framework <pt, tf> --model bert-base-cased --quantizepython convert_graph_to_onnx.py --framework <pt, tf> --model bert-base-cased --quantize
3 pip install onnxruntime-tools 
python -m onnxruntime_tools.optimizer_cli --input bert-base-cased.onnx --output bert-base-cased.onnx --model_type bert
4


