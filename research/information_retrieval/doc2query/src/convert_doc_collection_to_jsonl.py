# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os


def convert_collection(args):
    with open(args.output_path, "w", encoding="utf-8") as w:
        with open(args.collection_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                id, body = line.split("\t")
                output_dict = {"id": id, "contents": body}
                w.write(json.dumps(output_dict) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert MSMARCO tsv passage collection into jsonl files for Anserini."
    )
    parser.add_argument(
        "--collection_path", required=True, help="Path to MS MARCO tsv collection."
    )
    parser.add_argument("--output_path", required=True, help="Output filename.")
    args = parser.parse_args()
    convert_collection(args)
    print("Done!")
