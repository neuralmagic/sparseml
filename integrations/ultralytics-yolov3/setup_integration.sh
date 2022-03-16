#!/bin/bash

# Integration setup command to setup the folder so it is ready to train and sparsify models.
# Creates a yolov3 folder next to this script with all required dependencies from the ultralytics/yolov3 repository.
# Command: `bash setup_integration.sh`

git clone https://github.com/neuralmagic/yolov3.git
cd yolov3
git checkout release/0.11
pip install -r requirements.txt
