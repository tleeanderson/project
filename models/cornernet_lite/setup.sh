#!/bin/bash
GIT_PROJECT=CornerNet-Lite
PROJECT_NAME=CornerNet_Lite

if [ ! -d $PROJECT_NAME ]; then
  echo "Pulling down $GIT_PROJECT project..."
  git clone git@github.com:princeton-vl/$GIT_PROJECT.git $PROJECT_NAME
else
  echo "$PROJECT_NAME exists, will not pull from github"
fi

echo "setting up conda environment"
conda create --name CornerNet_Lite --file $PROJECT_NAME/conda_packagelist.txt --channel pytorch
eval "$(conda shell.bash hook)"
conda activate CornerNet_Lite

echo "compiling corner pooling layers"
cd $PROJECT_NAME/core/models/py_utils/_cpools/
python setup.py install --user

echo "Compiling non max suppresion"
cd -
cd $PROJECT_NAME/core/external
echo "current dir " $(pwd)
make

cd -
echo "moving model files"
declare -A DIRS_FILES
DIRS_FILES[CornerNet_Saccade]=CornerNet_Saccade_500000.pkl
DIRS_FILES[CornerNet_Squeeze]=CornerNet_Squeeze_500000.pkl
DIRS_FILES[CornerNet]=CornerNet_500000.pkl

ALL_EXIST="true"
for d in "${!DIRS_FILES[@]}"; do
  f=${DIRS_FILES[$d]}
  if [ ! -f ../../pretrained/$f ]; then
    echo $f "was not found, please execute tar -zxvf pretrained_models.tar.gz -C . in the project root"
    ALL_EXIST="false"
  fi
done

if [ "$ALL_EXIST" = true ]; then
  CACHE_PATH=cache/nnet
  MODEL_DIRS="CornerNet_Saccade CornerNet_Squeeze CornerNet"
  for d in "${!DIRS_FILES[@]}"; do
    f=${DIRS_FILES[$d]}
    mkdir -p $PROJECT_NAME/$CACHE_PATH/$d
    mv ../../pretrained/$f $PROJECT_NAME/$CACHE_PATH/$d/$f
  done
fi
