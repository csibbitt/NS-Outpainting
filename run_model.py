import random
import os
from glob import glob
import numpy as np
from PIL import Image
import tensorflow as tf
from model.generator import Generator
from model.loss import Loss
from dataset.build_dataset import input_hasher
from dataset.parse import parse_testset
import argparse
import math

parser = argparse.ArgumentParser(description='Model testing.')
# experiment
parser.add_argument('--date', type=str, default='1213')
parser.add_argument('--exp-index', type=int, default=1)
parser.add_argument('--f', action='store_true', default=False)

# gpu
parser.add_argument('--start-gpu', type=int, default=0)
parser.add_argument('--num-gpu', type=int, default=1)

# dataset
parser.add_argument('--trainset-path', type=str, default='./dataset/trainset.tfr')
parser.add_argument('--testset-path', type=str, default='./dataset/testset.tfr')
parser.add_argument('--trainset-length', type=int, default=5041)
parser.add_argument('--testset-length', type=int, default=2000)  # we flip every image in testset

# # training
parser.add_argument('--batch-size', type=int, default=20)
parser.add_argument('--weight-decay', type=float, default=0.00002)

parser.add_argument('--workers', type=int, default=2)

# checkpoint
parser.add_argument('--log-path', type=str, default='./logs/')
parser.add_argument('--checkpoint-path', type=str, default=None)
parser.add_argument('--resume-step', type=int, default=0)

# ***** Checkpoint handling
args = parser.parse_args("--batch-size 1 --testset-length 2000 --trainset-path ./dataset/trainset.tfr --testset-path ./dataset/testset.tfr --checkpoint-path checkpoint/-19939".split())

# prepare gpu
num_gpu = args.num_gpu
start_gpu = args.start_gpu
gpu_id = str(start_gpu)
for i in range(num_gpu - 1):
    gpu_id = gpu_id + ',' + str(start_gpu + i + 1)
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
args.batch_size_per_gpu = int(args.batch_size / args.num_gpu)

generator = Generator(args, name='model')
loss = Loss(args)

config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
config.graph_options.optimizer_options.global_jit_level = tf.compat.v1.OptimizerOptions.ON_1

def mainSession(buffer_size, img_callback, shuffle_flag, input_images, seed_hash, mix_strength):

    print("Start building model...")
    with tf.compat.v1.Session(config=config) as sess:
        with tf.device('/cpu:0'):

            testset = tf.compat.v1.data.TFRecordDataset(filenames=[args.testset_path])
            testset = testset.map(parse_testset, num_parallel_calls=args.workers)
            testset = testset.batch(args.batch_size).repeat()

            test_iterator = testset.make_one_shot_iterator()
            test_im = test_iterator.get_next()

            print('build model on gpu tower')
            models = []
            params = []
            for gpu_id in range(num_gpu):
                with tf.device('/gpu:%d' % gpu_id):
                    print('tower_%d' % gpu_id)
                    with tf.name_scope('tower_%d' % gpu_id):
                        with tf.compat.v1.variable_scope('cpu_variables', reuse=gpu_id > 0):

                            groundtruth = tf.compat.v1.placeholder(
                                tf.float32, [args.batch_size_per_gpu, 128, 256, 3], name='groundtruth')
                            left_gt = tf.slice(groundtruth, [0, 0, 0, 0], [args.batch_size_per_gpu, 128, 128, 3])

                            reconstruction = generator(left_gt)
                            right_recon = tf.slice(reconstruction, [0, 0, 128, 0], [args.batch_size_per_gpu, 128, 128, 3])

                            models.append((reconstruction, right_recon))
                            params.append(groundtruth)
            print('Done.')

            print('Start reducing towers on cpu...')
            reconstructions, right_recons = zip(*models)
            with tf.device('/gpu:0'):
                reconstructions = tf.concat(reconstructions, axis=0)
                right_recons = tf.concat(right_recons, axis=0)
            print('Done.')

            saver = tf.compat.v1.train.Saver(max_to_keep=5)
            if args.checkpoint_path is None:
                sess.run(tf.compat.v1.global_variables_initializer())
            else:
                print('Start loading checkpoint...')
                saver.restore(sess, args.checkpoint_path)
                print('Done.')

            print('run eval...')
            stitch_mask1 = np.ones((args.batch_size, 128, 128, 3))
            for i in range(128):
                stitch_mask1[:, :, i, :] = 1. / 127. * (127. - i)
            stitch_mask2 = stitch_mask1[:, :, ::-1, :]

            for _ in range(math.floor(args.testset_length / args.batch_size)):
                if len(input_images) == 0:
                    test_oris = sess.run([test_im])[0]
                    seed_hash.set('t_' + input_hasher(test_oris))
                else:
                    # Combo from build_dataset and train_model
                    image_bytes =  input_images[0].tobytes()
                    features = {'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))}
                    tf_features = tf.train.Features(feature=features)
                    tf_example = tf.train.Example(features=tf_features)
                    input_image_tensor = parse_testset(tf_example.SerializeToString(), shape=[1, 128, 128, 3])
                    input_image_tensor = tf.concat((input_image_tensor, tf.zeros([1, 128, 128, 3])), axis=2)
                    test_oris = sess.run([input_image_tensor])[0]
                origins1 = test_oris.copy()

                oris = None
                while not shuffle_flag.get():
                    inp_dict = {}
                    inp_dict = loss.feed_all_gpu(inp_dict, args.num_gpu, args.batch_size_per_gpu, test_oris, params)

                    if oris is None:
                        reconstruction_vals, prediction_vals = sess.run(
                            [reconstructions, right_recons],
                            feed_dict=inp_dict)

                        oris = reconstruction_vals
                        pred1 = oris[:, :, :128, :]
                        pred2 = oris[:, :, -128:, :]
                        gt = origins1[:, :, :128, :]
                        p1_m1 = np.concatenate((gt * stitch_mask1 + pred1 * stitch_mask2, pred2), axis=2)
                        img_callback(Image.fromarray((255. * (p1_m1[0] + 1) / 2.).astype(np.uint8)))
                        prediction_count = 2
                    else:
                        reconstruction_vals, prediction_vals = sess.run(
                            [reconstruction, right_recons],
                            feed_dict=inp_dict)
                        A = oris[:, :, -128:, :]
                        B = reconstruction_vals[:, :, :128, :]
                        C = A * stitch_mask1 + B * stitch_mask2
                        patch_count = np.shape(oris)[2] / 128
                        start_column = 128 if patch_count >= buffer_size.get() else 0
                        oris = np.concatenate((oris[:, :, start_column:-128, :], C, prediction_vals), axis=2)
                        img_callback(Image.fromarray((255. * (oris[0] + 1) / 2.).astype(np.uint8)))
                        prediction_count += 1

                    # Mix in original image to add style stability
                    ms = mix_strength.get() / 100
                    test_oris = np.concatenate(((prediction_vals +  ms * gt)/ (1 + ms), prediction_vals), axis=2)

                shuffle_flag.set(False)

def mainSessionMock(buffer_size, img_callback, shuffle_flag, input_images, seed_hash, mix_strength):
    eval_width = 128
    eval_height = 128
    file_list = glob('mock_images/endless*.jpg')
    while True:
        random.shuffle(file_list)
        mockImage = Image.open(file_list[0])
        seed_hash.set('m_' + input_hasher(mockImage))

        prediction_count = 0
        first = True
        while not shuffle_flag.get():
            if first:
                # The first frame is double-wide
                img_callback(mockImage.crop((0, 0, eval_width * 2, eval_height)))
                prediction_count = 2
                first = False
            else:
                prediction_count += 1
                start_column = 128 * (prediction_count - buffer_size.get()) if prediction_count >= buffer_size.get() else 0
                img_callback(mockImage.crop((start_column, 0, prediction_count * eval_width, eval_height)))
        shuffle_flag.set(False)