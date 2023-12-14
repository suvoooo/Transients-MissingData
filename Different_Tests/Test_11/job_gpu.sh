#!/bin/bash

source /lustre/ific.uv.es/ml/uv111/projects/PLAsTiCC_Challenge/Different_Tests/Test_11/setup.sh

ulimit -s unlimited

cd /lustre/ific.uv.es/ml/uv111/projects/PLAsTiCC_Challenge/Different_Tests/Test_11

python3 training_script.py > logs/training.log
