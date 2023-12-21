#!/bin/sh

if [ ! -f dataset/trainset.tfr ]; then
  wget "https://drive.google.com/uc?export=download&id=1XcL0guFyqhLns_HgkFBEKEpbUHQ1dA6U&confirm=yes" -O tf_scenery.zip
  unzip tf_scenery.zip
fi

export CUDA_HOME=/usr/local/cuda
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda
python train_model.py  --trainset-path ./dataset/trainset.tfr --testset-path ./dataset/testset.tfr
