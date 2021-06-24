#!/bin/bash

# Integration setup command to setup the folder so it is ready to train and sparsify models.
# Creates a transformers folder next to this script with all required dependencies from the huggingface/transformers repository.
# Command: `bash setup_integration.sh`

git clone https://github.com/neuralmagic/transformers.git
cd transformers
pip install -e .
pip install datasets
