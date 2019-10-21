#!/bin/bash

if [ ! -d ssd.pytorch ]; then
  echo "Pulling down ssd.pytorch project..."
  git clone https://github.com/amdegroot/ssd.pytorch  
else
  echo "ssd.pytorch exists, nothing to do, exiting"
fi

