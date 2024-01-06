CHECKPOINT="19939"

export CUDA_HOME=/usr/lib/cuda
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_HOME}"

python eval_model.py \
  --f \
  --date $(date +%Y%m%d) \
  --exp-index $(date +%H%M) \
  --trainset-path ./tf_dataset_new/trainset.tfr \
  --testset-path ./tf_dataset_new/testset.tfr \
  --log-path ./logs/ \
  --num-gpu 1 \
  --checkpoint-path checkpoint/-${CHECKPOINT} \
\
  --batch-size 4 \
  --testset-length 4
