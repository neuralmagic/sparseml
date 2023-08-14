python -m pyserini.search.faiss \
  --index wikipedia-dpr-single-nq-bf \
  --topics dpr-nq-test \
  --encoder $1 \
  --output runs/$1.trec \
  --batch-size 36 --threads 12

python -m pyserini.eval.convert_trec_run_to_dpr_retrieval_run \
  --index wikipedia-dpr \
  --topics dpr-nq-test \
  --input  runs/$1.trec  \
  --output  runs/$1.json


python -m pyserini.eval.evaluate_dpr_retrieval \
  --retrieval runs/$1.json\
  --topk 20 100 > runs/$1.metrics
