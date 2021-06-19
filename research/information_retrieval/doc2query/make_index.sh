python -m pyserini.index -collection JsonCollection \
                         -generator DefaultLuceneDocumentGenerator \
                         -threads 16 \
                         -input data/augmented_collection.jsonl \
                         -index data/indexes/msmarco_jsonl \
                         -storePositions -storeDocvectors -storeRaw
