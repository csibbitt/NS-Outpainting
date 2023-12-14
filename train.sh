export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda
python train_model.py  --trainset-path ./dataset/trainset.tfr --testset-path ./dataset/testset.tfr
