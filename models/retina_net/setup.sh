#!/bin/bash
GIT_PROJECT=pytorch-retinanet
PROJECT=Retina_Net

if [ ! -d $PROJECT ]; then
  echo "Pulling down $GIT_PROJECT project..."
  git clone git@github.com:yhenon/$GIT_PROJECT.git $PROJECT
else
  echo "$PROJECT exists, nothing to do, exiting"
fi

echo "building non max suppression"
cd $PROJECT/lib
bash build.sh
cd -
