import os
import json
import argparse

def convert_collection(args):
    with open(args.output_path, 'w', encoding='utf-8') as w:
        with open(args.collection_path, encoding='utf-8') as f:
            for i, line in enumerate(f):
                id, body = line.split('\t')
                output_dict = {'id': id, 'contents': body}
                w.write(json.dumps(output_dict) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert MSMARCO tsv passage collection into jsonl files for Anserini.')
    parser.add_argument('--collection_path', required=True, help='Path to MS MARCO tsv collection.')
    parser.add_argument('--output_path', required=True, help='Output filename.')
    args = parser.parse_args()
    convert_collection(args)
    print('Done!')
