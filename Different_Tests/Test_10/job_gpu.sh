#!/bin/bash

source /lustre/ific.uv.es/ml/uv111/projects/PLAsTiCC_Challenge/Different_Tests/Test_10/setup.sh

ulimit -s unlimited

cd /lustre/ific.uv.es/ml/uv111/projects/PLAsTiCC_Challenge/Different_Tests/Test_10

python3 training_script.py > logs/training.log
