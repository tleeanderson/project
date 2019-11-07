#!/bin/bash
GIT_PROJECT=CornerNet-Lite
PROJECT_NAME=CornerNet_Lite

if [ ! -d $PROJECT_NAME ]; then
  echo "Pulling down $GIT_PROJECT project..."
  git clone git@github.com:princeton-vl/$GIT_PROJECT.git $PROJECT_NAME
else
  echo "$PROJECT_NAME exists, nothing to do, exiting"
fi

echo "setting up conda environment"
conda create --name CornerNet_Lite --file $PROJECT_NAME/conda_packagelist.txt --channel pytorch
eval "$(conda shell.bash hook)"
source activate CornerNet_Lite

echo "compiling corner pooling layers"
cd $PROJECT_NAME/core/models/py_utils/_cpools/
python setup.py install --user

echo "Compiling non max suppresion"
cd -
cd $PROJECT_NAME/core/external
make
