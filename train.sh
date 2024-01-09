#!/bin/sh

CHECKPOINT=""
export CUDA_HOME=/usr/lib/cuda
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_HOME}"
#export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/share/miniconda/lib/ # Genesis

if [ ! -f tf_dataset_new/trainset.tfr ]; then
  wget "https://drive.google.com/uc?export=download&id=1XcL0guFyqhLns_HgkFBEKEpbUHQ1dA6U&confirm=yes" -O tf_scenery.zip
  unzip tf_scenery.zip
fi

if [ ! -z "${CHECKPOINT}" ]; then CKPT_OPTS=" --checkpoint-path ./checkpoint/-${CHECKPOINT}  --resume-step ${CHECKPOINT}"; fi

python train_model.py \
  --f \
  --date $(date +%Y%m%d) \
  --exp-index $(date +%H%M) \
  --trainset-path ./tf_dataset_new/trainset.tfr \
  --testset-path ./tf_dataset_new/testset.tfr \
  --log-path ./logs/ \
  --num-gpu 1 \
\
  --deterministic-seed 1 \
  --trainset-length 4 \
  --testset-length 4 \
  --batch-size 2 \
  --epoch 3 \
  --warmup-steps 2 \
  # --load-v2-checkpoint \
  # --checkpoint-path logs/20240108/920/checkpoint/ckpt-25 \
  ${CKPT_OPTS}
