#!/bin/bash

# Integration setup command to setup the folder so it is ready to train and sparsify models.
# Creates a pytorch-image-models folder next to this script with all required dependencies from the rwightman/pytorch-image-models repository.
# Command: `bash setup_integration.sh`

git clone https://github.com/neuralmagic/pytorch-image-models.git
cd pytorch-image-models
git checkout release/1.1
pip install -r requirements.txt
pip install sparseml[torch]
