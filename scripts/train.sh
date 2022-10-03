#!/bin/bash

# activate environment
source /home/al826/rds/hpc-work/envs/torch1.12/bin/activate

#load any enviornment variables needed
source ~/.bashrc

TOKENIZERS_PARALLELISM=false

python ../run_train.py $@
