CHECKPOINT="19939"

export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda
python eval_model.py --trainset-path ./dataset/trainset.tfr --testset-path ./dataset/testset.tfr --checkpoint-path model/-${CHECKPOINT}
