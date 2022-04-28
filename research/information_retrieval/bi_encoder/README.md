# Transfer Encoder

python bi_encoder.py --model_name_or_path bert-base-uncased --mimic_model_name_or_path facebook/dpr-question_encoder-single-nq-base --num_train_epochs 10 --learning_rate 5e-5 --recipe downstream.yaml  --max_length 32 --per_device_train_batch_size 128 --output_dir biencoder-facebook-dpr-single-nq-base-gmpdownstream90-start-bert-base-uncased --seed 32 --dataset_name Tevatron/wikipedia-nq

# Evaluate Quality

sh eval_encoder.sh biencoder-facebook-dpr-single-nq-base-gmpdownstream90





Expected  finetuned bert-base-uncased
Top20	accuracy: 0.8006
Top100	accuracy: 0.8609


bert-base-uncased
Top20	accuracy: 0.017451523545706372
Top100	accuracy: 0.09889196675900278

80 block
100%|█████████████████████████████████████████████████████████████████████████████████████████| 3610/3610 [00:00<00:00, 119430.80it/s]
Top20	accuracy: 0.21357340720221607
Top100	accuracy: 0.3631578947368421


90 
Top20	accuracy: 0.06703601108033241
Top100	accuracy: 0.1587257617728532

https://github.com/castorini/pyserini/blob/master/docs/experiments-dpr.md
