CHECKPOINT="19939"

export CUDA_HOME=/usr/lib/cuda
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_HOME}"

python eval_model.py --trainset-path ./dataset/trainset.tfr --testset-path ./dataset/testset.tfr --checkpoint-path checkpoint/-${CHECKPOINT}
