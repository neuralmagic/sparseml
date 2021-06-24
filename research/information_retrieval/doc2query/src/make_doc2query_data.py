import argparse
import os
import json
def load_qid2query(filename):
    qid2query = {}
    with open(filename, 'r') as f:
        for l in f:
            l = l.strip().split('\t')
            qid2query[int(l[0])] = l[1]
    return qid2query

def load_qrels(filename, collection, qid2query):
    qrels = {}
    with open(filename, 'r') as f:
        for l in f:
            l = l.strip().split('\t')
            qrels[qid2query[int(l[0])]] = collection[int(l[2])]
    return qrels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--collection_file",
        default="collection.tsv",
        type=str,
        help="The msmarco passage collection file",
    )
    parser.add_argument(
        "--query_file",
        type=str,
        default="queries.tsv",
        help="Qid to query for all msmarco queries",
    )
    parser.add_argument(
        "--train_qrel_file",
        type=str,
        default="qrels.train.tsv",
        help="The input file in TSV form of doc2query",
    )
    parser.add_argument(
        "--dev_qrel_file",
        type=str,
        default="qrels.dev.tsv",
        help="The input file in TSV form of doc2query",
    )
    parser.add_argument(
        "--output_file_prefix",
        type=str,
        default="doc_query_",
        help="The input file in TSV form of doc2query",
    )
    args = parser.parse_args()
    collection = load_qid2query(args.collection_file)
    qid2query = load_qid2query(args.query_file)
    train_qrels = load_qrels(args.train_qrel_file, collection, qid2query)
    dev_qrels = load_qrels(args.dev_qrel_file, collection, qid2query)
    with open(args.output_file_prefix+"train.json",'w') as w:  
        for qrel in train_qrels:
            w.write("{}\n".format(json.dumps({"input":train_qrels[qrel], "target":qrel})))
    with open(args.output_file_prefix+"dev.json",'w') as w:    
        for qrel in dev_qrels:
            w.write("{}\n".format(json.dumps({"input":dev_qrels[qrel], "target":qrel})))


if __name__ == "__main__":
    main()
