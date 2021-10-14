#!/bin/bash
export PYTHONWARNINGS="ignore"

python /cephfs/jianyu/ActiveStereoNet/tools/train_activestereo.py \
--config /cephfs/jianyu/GANPSMFeature/configs/remote_train_gan.yaml \
--log_freq 1 --logdir /cephfs/jianyu/eval/activestereo_train
