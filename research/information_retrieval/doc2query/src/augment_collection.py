import argparse
import os
import json

import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

def load_qid2query(filename):
    qid2query = {}
    with open(filename, 'r') as f:
        for l in f:
            l = l.strip().split('\t')
            qid2query[int(l[0])] = l[1]
    return qid2query


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--collection_file",
        default="data/collection.tsv",
        type=str,
        help="The msmarco passage collection file",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="Doc2Query predictions",
    )
    parser.add_argument(
        "--augmented_collection_file",
        type=str,
        default="data/augmented_collection.jsonl",
        help="The output_file for augmented doc 2 query index",
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=3,
        help="number of queries to generate per passage",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=32,
        help="length of document queries",
    )
    parser.add_argument(
        '--no_cuda',
        action="store_true",
        help="Use this to not use cuda")
    args = parser.parse_args()
    print("Loading collection")
    collection = load_qid2query(args.collection_file)
    print("Collection loaded")
    device='cuda'
    if args.no_cuda:
        device='cpu'

    print("Loading model")
    config = AutoConfig.from_pretrained(args.model_name_or_path,)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    model.to(device)
    model.resize_token_embeddings(len(tokenizer))
    print("Model Loaded")
    print("Augmenting passages")
    augmentations = 0
    #TODO Introduce batching at inference time as right now runs 1 by 1
    with open(args.augmented_collection_file, 'w') as w:
        for doc_id in collection:
            if augmentations % 5000 == 0:
                print("{} passages augmented".format(augmentations))
            document_text = collection[doc_id]
            input_ids = tokenizer.encode(document_text, return_tensors='pt').to(device)
            outputs = model.generate(
                input_ids=input_ids,
                max_length=args.max_length,
                do_sample=True,
                top_k=10,
                num_return_sequences=args.beam_size)
            query_augment = ''
            for i in range(args.beam_size):
                query_augment += ' '
                query_augment += tokenizer.decode(outputs[i], skip_special_tokens=True)
            output_dict = {'id': doc_id, 'contents': document_text + query_augment}
            w.write(json.dumps(output_dict) + '\n')
            augmentations += 1
        
if __name__ == "__main__":
    main()

