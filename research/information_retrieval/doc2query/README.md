# Doc2Query Compressed

Author: @spacemanidol

Doc2query introduced a simple and direct method to integrate neural information retrieval in context of tradition keyword search. Instead of introducing a neural ranking engine at query time neural methods are moved to index generation time. 
A sequence to sequence is trained with the input being passages(short context windows) and the target being the relevant query. Since the MSMARCO coprus features over 500,000 relevant passages methods like T5 can be leveraged. Unfortunatley, without compression existing T5 takes the index generation from 10 minutes(16 threads on a 14 core Intel(R) Xeon(R) Gold 5120 CPU @ 2.20GHz) to > using 4 16 GB V100
## Results

| Method       | Sparsity | MRR @10 MSMARCO Dev | Latency(s) per 1000 queries | Index Generation (S)|Citation        |
|--------------|----------|---------------------|-----------------------------|---------------------|----------------|
|BM25(Anserini)|0         |0.1874               |79.85                        |00:10:16


### Baseline
Download the data
```sh
cd data
wget https://msmarco.blob.core.windows.net/msmarcoranking/collectionandqueries.tar.gz
tar -xzvf collectionandqueries.tar.gz
rm collectionandqueries.tar.gz
cat queries.dev.tsv queries.train.tsv queries.eval.tsv > queries.tsv
```

To format the collections file, build simple index, run on msmarco dev set and evaluate which should produce outpu
```
mkdir data/base_collection
python src/convert_doc_collection_to_jsonl.py --collection_path data/collection.tsv --output_path data/base_collection/collection
python -m pyserini.index -collection JsonCollection -generator DefaultLuceneDocumentGenerator \
 -threads 16 -input data/base_collection \
 -index indexes/msmarco-passage-baseline -storePositions -storeDocvectors -storeRaw
python -m pyserini.search --topics data/queries.dev.small.tsv \
 --index indexes/msmarco-passage-baseline \
 --output runs/run.msmarco-passage.bm25baseline.tsv \
 --bm25 --output-format msmarco --hits 1000 --k1 0.82 --b 0.68
python src/msmarco_passage_eval.py data/qrels.dev.small.tsv runs/run.msmarco-passage.bm25baseline.tsv
#####################
MRR @10: 0.18741227770955543
QueriesRanked: 6980
#####################
```

### Doc2Query

Format the data for training

```sh
 python src/make_doc2query_data.py --collection_file data/collection.tsv --query_file data.queries.tsv --train_qrel_file data/qrels.train.tsv --dev_qrel_file data/qrels.dev.tsv --output_file_prefix data/doc_query_
