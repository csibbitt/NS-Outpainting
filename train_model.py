import os
from glob import glob
import numpy as np
from PIL import Image
import tensorflow as tf
from model.generator import Generator
from model.loss import Loss
from dataset.parse import parse_trainset, parse_testset
import argparse
import time
import sys

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Timer():
    def __init__(self, label):
        self.label = label
        self.start = time.perf_counter()
        self.lapstart = self.start

    def stop(self):
        print('{} taking : {}s'.format(self.label, int(time.perf_counter() - self.start)))

    def lap(self):
        now = time.perf_counter()
        print('{} lap taking : {}s'.format(self.label, int(now - self.lapstart)))
        self.lapstart = now

    def reset(self):
        self.start = time.perf_counter()
        self.lap = self.start



parser = argparse.ArgumentParser(description='Model training.')
# experiment
parser.add_argument('--date', type=str, default='1214')
parser.add_argument('--exp-index', type=int, default=2)
parser.add_argument('--f', action='store_true', default=False)

# gpu
parser.add_argument('--start-gpu', type=int, default=0)
parser.add_argument('--num-gpu', type=int, default=1)

# dataset
parser.add_argument('--trainset-path', type=str, default='./dataset/trainset.tfr')
parser.add_argument('--testset-path', type=str, default='./dataset/testset.tfr')
parser.add_argument('--trainset-length', type=int, default=5041)
parser.add_argument('--testset-length', type=int, default=2000)  # we flip every image in testset

# training
parser.add_argument('--base-lr', type=float, default=0.0001)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--weight-decay', type=float, default=0.00002)
parser.add_argument('--epoch', type=int, default=1500)
parser.add_argument('--lr-decay-epoch', type=int, default=1000)
parser.add_argument('--critic-steps', type=int, default=3)
parser.add_argument('--warmup-steps', type=int, default=1000)
parser.add_argument('--workers', type=int, default=2)
parser.add_argument('--clip-gradient', action='store_true', default=False)
parser.add_argument('--clip-gradient-value', type=float, default=0.1)


# model
parser.add_argument('--beta', type=float, default=0.9)
parser.add_argument('--lambda-gp', type=float, default=10)
parser.add_argument('--lambda-rec', type=float, default=0.998)

# checkpoint
parser.add_argument('--log-path', type=str, default='./logs/')
parser.add_argument('--checkpoint-path', type=str, default=None)

# debugging
parser.add_argument('--deterministic-seed', type=int, default=0)
parser.add_argument('--check-numerics', action='store_true', default=False)

args = parser.parse_args()

# prepare path
base_path = args.log_path
exp_date = args.date
if exp_date is None:
    print('Exp date error!')
    import sys
    sys.exit()
exp_name = exp_date + '/' + str(args.exp_index)
print("Start Exp:", exp_name)
output_path = base_path + exp_name + '/'
ckpt_path = output_path + 'checkpoint/'
tensorboard_path = output_path + 'log/'
result_path = output_path + 'results/'

if not os.path.exists(tensorboard_path):
    os.makedirs(tensorboard_path)
if not os.path.exists(result_path):
    os.makedirs(result_path)
elif not args.f:
    if args.checkpoint_path is None:
        print('Exp exist!')
        import sys
        sys.exit()
else:
    import shutil
    shutil.rmtree(tensorboard_path)
    os.makedirs(tensorboard_path)

# prepare gpu
num_gpu = args.num_gpu
start_gpu = args.start_gpu
gpu_id = str(start_gpu)
for i in range(num_gpu - 1):
    gpu_id = gpu_id + ',' + str(start_gpu + i + 1)
args.batch_size_per_gpu = int(args.batch_size / args.num_gpu)

print("Start building model...")
writer = tf.summary.create_file_writer(tensorboard_path)
writer.init()

if args.deterministic_seed != 0:
    tf.keras.utils.set_random_seed(args.deterministic_seed)
    tf.random.set_seed(args.deterministic_seed)
    tf.config.experimental.enable_op_determinism()

with tf.device('/cpu:0'):
    trainset = tf.data.TFRecordDataset(filenames=[args.trainset_path])
    trainset = trainset.shuffle(args.trainset_length)
    trainset = trainset.map(parse_trainset, num_parallel_calls=tf.data.AUTOTUNE)
    trainset = trainset.batch(args.batch_size, drop_remainder=True).prefetch(2).repeat()
    train_im = iter(trainset)

    testset = tf.data.TFRecordDataset(filenames=[args.testset_path])
    testset = testset.map(parse_testset, num_parallel_calls=tf.data.AUTOTUNE)
    testset = testset.batch(args.batch_size, drop_remainder=True).prefetch(2).repeat()
    test_im = iter(testset)

generator = Generator(args)
loss = Loss(args)

learning_rate = tf.Variable(args.base_lr, dtype=tf.float32, shape=[])
lambda_rec =  tf.Variable(args.lambda_rec, dtype=tf.float32, shape=[])

G_opt = tf.keras.optimizers.Adam(
    learning_rate=learning_rate, beta_1=0.5, beta_2=0.9, epsilon=1e-08)
D_opt = tf.keras.optimizers.Adam(
    learning_rate=learning_rate, beta_1=0.5, beta_2=0.9, epsilon=1e-08)

step = tf.Variable(0, dtype=tf.int64, trainable=False)
ckpt_epoch = tf.Variable(0, dtype=tf.int64, trainable=False)
ckpt = tf.train.Checkpoint(step=step,
                            epoch=ckpt_epoch,
                            G_opt=G_opt,
                            D_opt=D_opt,
                            generator=generator,
                            discrim_g=loss.discrim_g,
                            discrim_l=loss.discrim_l)
ckpt_manager = tf.train.CheckpointManager(ckpt, directory=ckpt_path, max_to_keep=5)


apply_grads_g = tf.Variable(False, dtype=tf.bool)
apply_grads_d = tf.Variable(False, dtype=tf.bool)

@tf.function(jit_compile=True)
def fwd(groundtruth):
    with tf.GradientTape() as g_G, tf.GradientTape() as g_D:
        g_G.watch(groundtruth)
        g_D.watch(groundtruth)

        left_gt = tf.slice(groundtruth, [0, 0, 0, 0], [args.batch_size_per_gpu, 128, 128, 3])
        reconstruction = generator(left_gt)

        loss_rec = loss.masked_reconstruction_loss(groundtruth, reconstruction)  #** Could skip this when only training D()
        loss_adv_G, loss_adv_D = loss.global_and_local_adv_loss(groundtruth, reconstruction) #** Could skip this during G() warmup

        loss_G = loss_adv_G * (1 - lambda_rec) + loss_rec * lambda_rec + tf.reduce_sum(generator.losses)
        loss_D = loss_adv_D

        var_G = generator.trainable_variables
        var_D = loss.discrim_l.trainable_variables + loss.discrim_g.trainable_variables

    grad_g = G_opt.compute_gradients(loss_G, var_G, tape=g_G)
    grad_d = D_opt.compute_gradients(loss_D, var_D, tape=g_D)

    if(apply_grads_g.value()):
        G_opt.apply_gradients(grad_g)
    if(apply_grads_d.value()):
        D_opt.apply_gradients(grad_d)

    return loss_G, loss_adv_G, loss_D, loss_rec, grad_g, grad_d, reconstruction

iters = 0
if args.checkpoint_path is not None:
    print('Start restore checkpoint...')
    fwd(tf.zeros([args.batch_size_per_gpu, 128, 256, 3])) # Run a forward pass to construct all vars for .assert_consumed()
    status = ckpt.restore(args.checkpoint_path).assert_consumed()
    iters = step.numpy()
    print('Done.')

if args.check_numerics:
    tf.debugging.enable_check_numerics()

print('Start training...')
for epoch in range(ckpt_epoch.numpy(), args.epoch):
    etimer = Timer(f'epoch {epoch}')
    if epoch > args.lr_decay_epoch:
        learning_rate.assign(args.base_lr / 10)

    itimer = Timer('iter')
    for start, end in zip(
            range(0, args.trainset_length, args.batch_size),
            range(args.batch_size, args.trainset_length, args.batch_size)):

        if iters == 0:
            gtimer = Timer('G warmup')
            print('Start pretraining G!')
            lambda_rec.assign(1.)
            apply_grads_g.assign(True)
            for t in range(args.warmup_steps):
                if t % 20 == 0:
                    print("Step:", t)
                images = train_im.get_next()

                loss_G, loss_adv_G, loss_D, loss_rec, grad_g, grad_d, reconstruction = fwd(images)
            gtimer.stop()
            print('Pre-train G Done!')

        lambda_rec.assign(args.lambda_rec)

        if (iters < 25) or iters % 500 == 0:
            n_cir = 30
        else:
            n_cir = args.critic_steps

        # Train D
        apply_grads_g.assign(False)
        apply_grads_d.assign(True)
        for t in range(n_cir):
            images =  train_im.get_next()
            if len(images) < args.batch_size:
                images =  train_im.get_next()

            loss_G, loss_adv_G, loss_D, loss_rec, grad_g, grad_d, reconstruction = fwd(images)

        # Train G
        apply_grads_g.assign(True)
        apply_grads_d.assign(False)
        loss_G, loss_adv_G, loss_D, loss_rec, grad_g, grad_d, reconstruction = fwd(images)

        if iters % 25 == 0:
            print("Iter:", iters, 'loss_g:', loss_G.numpy(), 'loss_d:', loss_D.numpy(), 'loss_adv_g:', loss_adv_G.numpy(), 'loss_rec:', loss_rec.numpy())
            itimer.lap()
            if np.isnan(loss_G.numpy()):
                print("NaN detected!!");
                sys.exit()

            with writer.as_default(step=step):
                tf.summary.scalar('loss/g', loss_G)
                tf.summary.scalar('loss/d', loss_D)
                tf.summary.scalar('loss/ag', loss_adv_G)
                tf.summary.scalar('loss/rec', loss_rec)
                tf.summary.image('groundtruth', tf.dtypes.cast(((images + 1) * 255. / 2.), tf.uint8), max_outputs=2)
                tf.summary.image('reconstruction', tf.dtypes.cast(((reconstruction + 1) * 255. / 2.), tf.uint8), max_outputs=2)
            writer.flush()

        iters += 1
        step.assign_add(1)
    ckpt_epoch.assign_add(1)
    ckpt_manager.save()


    # testing
    if epoch > 0:
        ii = 0
        g_vals = 0
        d_vals = 0
        ag_vals = 0
        n_batchs = 0
        apply_grads_g.assign(False)
        apply_grads_d.assign(False)
        eval_timer = Timer('eval')
        for _ in range(int(args.testset_length / args.batch_size)):
            test_oris = test_im.get_next()

            g_val, ag_val, d_val, loss_rec, grad_g, grad_d, reconstruction_vals = fwd(test_oris)

            g_vals += g_val
            d_vals += d_val
            ag_vals += ag_val
            n_batchs += 1

            # Save test results
            if epoch % 100 == 0:
                rtimer = Timer('results')
                for rec_val, test_ori in zip(reconstruction_vals, test_oris):
                    rec_hid = (255. * (rec_val.numpy() + 1) /
                                2.).astype(np.uint8)
                    test_ori = (255. * (test_ori.numpy() + 1) /
                                2.).astype(np.uint8)
                    Image.fromarray(rec_hid).save(os.path.join(
                        result_path, 'img_' + str(ii) + '.' + str(int(iters / 100)) + '.jpg'))
                    if epoch == 0:
                        Image.fromarray(test_ori).save(
                            os.path.join(result_path, 'img_' + str(ii) + '.' + str(int(iters / 100)) + '.ori.jpg'))
                    ii += 1
                rtimer.stop()
        g_vals /= n_batchs
        d_vals /= n_batchs
        ag_vals /= n_batchs

        print("=========================================================================")
        print('loss_g:', g_val.numpy(), 'loss_d:', d_val.numpy(), 'loss_adv_g:', ag_val.numpy())
        print("=========================================================================")

        with writer.as_default(step=step):
            tf.summary.scalar('eval/g', g_vals)
            tf.summary.scalar('eval/d', d_vals)
            tf.summary.scalar('eval/ag', ag_vals)
        writer.flush()

        if np.isnan(g_val):
            print("NaN detected!!")
            sys.exit()
        eval_timer.stop()
    etimer.stop()

#cProfile.run('main()')
