#!/bin/sh

CHECKPOINT=24963

export CUDA_HOME=/usr/local/cuda
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda

if [ ! -f tf_dataset_new/trainset.tfr ]; then
  wget "https://drive.google.com/uc?export=download&id=1XcL0guFyqhLns_HgkFBEKEpbUHQ1dA6U&confirm=yes" -O tf_scenery.zip
  unzip tf_scenery.zip
fi

python train_model.py\
  --trainset-path ./tf_dataset_new/trainset.tfr \
  --testset-path ./tf_dataset_new/testset.tfr \
  --log-path ./logs/ \
  --checkpoint-path "./drive/MyDrive/v2_NS-Outpainting/logs/1215/2/models/-${CHECKPOINT}" \
  --resume-step "${CHECKPOINT}"