#!/bin/bash
PROJECT=CornerNet

if [ ! -d $PROJECT ]; then
  echo "Pulling down $PROJECT project..."
  git clone git@github.com:princeton-vl/$PROJECT.git
else
  echo "$PROJECT exists, nothing to do, exiting"
fi

