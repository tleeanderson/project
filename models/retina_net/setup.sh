#!/bin/bash
GIT_PROJECT=keras-retinanet
PROJECT=keras_retina_net

if [ ! -d $PROJECT ]; then
  echo "Pulling down $GIT_PROJECT project..."
  git clone git@github.com:fizyr/$GIT_PROJECT.git $PROJECT
else
  echo "$PROJECT_NAME exists, will not pull from github"
fi

cd $PROJECT
pip3 install . --user
pip3 install keras-resnet --user
