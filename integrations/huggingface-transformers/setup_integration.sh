#!/bin/bash

# Integration setup command to setup the folder so it is ready to train and sparsify models.
# Creates a transformers folder next to this script with all required dependencies from the huggingface/transformers repository.
# Command: `bash setup_integration.sh`

git clone https://github.com/neuralmagic/transformers.git
cd transformers
git checkout release/0.8
pip install "torch<1.9"
pip install -e .
pip install sparseml[torch] datasets seqeval
