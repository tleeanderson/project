#!/bin/bash

eval "$(/anaconda/bin/conda shell.bash hook)"
conda init
conda activate $1

ls /CornerNet-Lite/data/coco/PythonAPI/pycocotools
python --version

python train.py $2
