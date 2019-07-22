#!/usr/bin/env bash

set -u   # crash on missing env variables
set -e   # stop on any error
set -x   # print what we are doing

if [ "$#" -ne 2 ]; then
  echo "Usage model_name json_input" >&2
  exit 1
fi

python ./run_mirror_objectstore.py tmp/objectstore

export MODEL_DIR=tmp/objectstore/automation/models/$1
JSON_INPUT=tmp/objectstore/automation/prediction/$2
python ./run_prediction.py $JSON_INPUT
