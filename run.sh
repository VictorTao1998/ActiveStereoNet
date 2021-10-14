#!/bin/bash
export PYTHONWARNINGS="ignore"

python /cephfs/jianyu/ActiveStereoNet/train_activestereo.py \
--config /cephfs/jianyu/ActiveStereoNet/configs/remote_train_gan.yaml \
--log_freq 100 --logdir /cephfs/jianyu/eval/activestereo_train
