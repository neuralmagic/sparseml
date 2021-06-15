wget https://msmarco.blob.core.windows.net/msmarcoranking/qrels.train.tsv
wget https://msmarco.blob.core.windows.net/msmarcoranking/qrels.dev.tsv
wget https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz
tar -xzvf collection.tar.gz
rm collection.tar.gz
wget https://msmarco.blob.core.windows.net/msmarcoranking/queries.tar.gz
tar -xzvf queries.tar.gz
cat queries.* > queries.tsv
rm queries.dev.tsv queries.eval.tsv queries.train.tsv