import argparse
import os
import json

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_file",
        default="data/hf_doc_query_pairs_train.json",
        type=str,
        help="The output file for HF compatible doc2query generation",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="data/doc_query_pairs.train.tsv",
        help="The input file in TSV form of doc2query",
    )
    args = parser.parse_args()
    with open(args.input_file,'r') as f:
        with open(args.output_file,'w') as w:
            for l in f:
                l = l.strip().split('\t')
                w.write("{}\n".format(json.dumps({"input":l[0], "target":l[1]})))


if __name__ == "__main__":
    main()
