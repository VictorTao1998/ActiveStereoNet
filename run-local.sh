#!/bin/bash
export PYTHONWARNINGS="ignore"

python /code/train_activestereo.py \
--config /code/disparity/configs/local_train_gan.yaml \
--log_freq 1 --logdir /data/eval/activestereo_train
