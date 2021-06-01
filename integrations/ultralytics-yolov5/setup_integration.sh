#!/bin/bash

# Integration setup command to setup the folder so it is ready to train and sparsify models.
# Creates a yolov5 folder next to this script with all required dependencies from the ultralytics/yolov5 repository.
# Command: `bash setup_integration.sh`

git clone https://github.com/neuralmagic/yolov5.git
pip install -r yolov5/requirements.txt
