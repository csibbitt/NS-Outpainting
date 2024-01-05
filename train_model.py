import os
from glob import glob
import numpy as np
from PIL import Image
import tensorflow as tf
from modeling.model import Model
from modeling.loss import Loss
from dataset.parse import parse_trainset, parse_testset
import argparse
import time
import cProfile

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


# modeling
parser.add_argument('--beta', type=float, default=0.9)
parser.add_argument('--lambda-gp', type=float, default=10)
parser.add_argument('--lambda-rec', type=float, default=0.998)

# checkpoint
parser.add_argument('--log-path', type=str, default='./logs/')
parser.add_argument('--checkpoint-path', type=str, default=None)
parser.add_argument('--resume-step', type=int, default=0)

# determinism
parser.add_argument('--deterministic-seed', type=int, default=0)
parser.add_argument('--dump-vars', action='store_true', default=False)




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
model_path = output_path + 'models/'
tensorboard_path = output_path + 'log/'
result_path = output_path + 'results/'

if not os.path.exists(model_path):
    os.makedirs(model_path)
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
    shutil.rmtree(model_path)
    os.makedirs(model_path)
    shutil.rmtree(tensorboard_path)
    os.makedirs(tensorboard_path)

# prepare gpu
num_gpu = args.num_gpu
start_gpu = args.start_gpu
gpu_id = str(start_gpu)
for i in range(num_gpu - 1):
    gpu_id = gpu_id + ',' + str(start_gpu + i + 1)
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
args.batch_size_per_gpu = int(args.batch_size / args.num_gpu)



#def main():

model = Model(args)
loss = Loss(args)

config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
config.graph_options.optimizer_options.global_jit_level = tf.compat.v1.OptimizerOptions.ON_1

print("Start building model...")
with tf.compat.v1.Session(config=config) as sess:

    if args.deterministic_seed != 0:
        tf.compat.v1.random.set_random_seed(1)
        tf.keras.utils.set_random_seed(1)
        tf.config.experimental.enable_op_determinism()
        tf.random.set_seed(1)

    with tf.device('/cpu:0'):

        btimer = Timer('building')
        learning_rate = tf.compat.v1.placeholder(tf.float32, [])
        lambda_rec = tf.compat.v1.placeholder(tf.float32, [])

        train_op_G = tf.compat.v1.train.AdamOptimizer(
            learning_rate=learning_rate, beta1=0.5, beta2=0.9)
        train_op_D = tf.compat.v1.train.AdamOptimizer(
            learning_rate=learning_rate, beta1=0.5, beta2=0.9)


        trainset = tf.compat.v1.data.TFRecordDataset(filenames=[args.trainset_path])
        trainset = trainset.shuffle(args.trainset_length)
        trainset = trainset.map(parse_trainset, num_parallel_calls=args.workers)
        trainset = trainset.batch(args.batch_size).repeat()

        train_iterator = trainset.make_one_shot_iterator()
        train_im = train_iterator.get_next()

        testset =tf.compat.v1.data.TFRecordDataset(filenames=[args.testset_path])
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


                        reconstruction_ori, reconstruction = model(left_gt)
                        right_recon = tf.slice(reconstruction, [0, 0, 128, 0], [args.batch_size_per_gpu, 128, 128, 3])

                        loss_rec = loss.masked_reconstruction_loss(groundtruth, reconstruction)
                        loss_adv_G, loss_adv_D = loss.global_and_local_adv_loss(model, groundtruth, reconstruction)
# ***** Check these in debugger
                        reg_losses = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)
                        loss_G = loss_adv_G * (1 - lambda_rec) + loss_rec * lambda_rec + sum(reg_losses)
                        loss_D = loss_adv_D

                        var_G = list(filter(lambda x: x.name.find(
                            'cpu_variables/GEN') != -1, tf.compat.v1.trainable_variables()))
                        var_D = list(filter(lambda x: x.name.find(
                            'cpu_variables/DIS') != -1, tf.compat.v1.trainable_variables()))

# ***** Check these in debugger
                        grad_g = train_op_G.compute_gradients(
                            loss_G, var_list=var_G)
                        grad_d = train_op_D.compute_gradients(
                            loss_D, var_list=var_D)

                        models.append((grad_g, grad_d, loss_G, loss_D, loss_adv_G, loss_rec, reconstruction))
                        params.append(groundtruth)

        print('Done.')
        btimer.stop()

        print('Start reducing towers on gpu...')

        grad_gs, grad_ds, loss_Gs, loss_Ds, loss_adv_Gs, loss_recs, reconstructions = zip(*models)
        groundtruths = params

        with tf.device('/gpu:0'):
            aver_loss_g = tf.reduce_mean(input_tensor=loss_Gs)
            aver_loss_d = tf.reduce_mean(input_tensor=loss_Ds)
            aver_loss_ag = tf.reduce_mean(input_tensor=loss_adv_Gs)
            aver_loss_rec = tf.reduce_mean(input_tensor=loss_recs)

            train_op_G = train_op_G.apply_gradients(
                loss.average_gradients(grad_gs))
            train_op_D = train_op_D.apply_gradients(
                loss.average_gradients(grad_ds))

            groundtruths = tf.concat(groundtruths, axis=0)
            reconstructions = tf.concat(reconstructions, axis=0)

            tf.compat.v1.summary.scalar('loss_g', aver_loss_g)
            tf.compat.v1.summary.scalar('loss_d', aver_loss_d)
            tf.compat.v1.summary.scalar('loss_ag', aver_loss_ag)
            tf.compat.v1.summary.scalar('loss_rec', aver_loss_rec)
            tf.compat.v1.summary.image('groundtruth', groundtruths, 2)
            tf.compat.v1.summary.image('reconstruction', reconstructions, 2)

            merged = tf.compat.v1.summary.merge_all()
            writer = tf.compat.v1.summary.FileWriter(tensorboard_path, sess.graph)

        print('Done.')


        iters = 0
        saver = tf.compat.v1.train.Saver(max_to_keep=3)
        if args.checkpoint_path is None:
            sess.run(tf.compat.v1.global_variables_initializer())
        else:
            print('Start loading checkpoint...')
            saver.restore(sess, args.checkpoint_path)
            iters = args.resume_step
            print('Done.')

        if args.dump_vars:
            import tf_slim as slim
            model_vars = tf.compat.v1.trainable_variables()
            slim.model_analyzer.analyze_vars(model_vars, print_info=True)

        print('Start training...')

        for epoch in range(args.epoch):
            etimer = Timer(f'epoch {epoch}')
            if epoch > args.lr_decay_epoch:
                learning_rate_val = args.base_lr / 10
            else:
                learning_rate_val = args.base_lr

            itimer = Timer('iter')
            for start, end in zip(
                    range(0, args.trainset_length, args.batch_size),
                    range(args.batch_size, args.trainset_length, args.batch_size)):

                if iters == 0 and args.checkpoint_path is None:
                    print('Start pretraining G!')
                    for t in range(args.warmup_steps):
                        if t % 20 == 0:
                            print("Step:", t)
                        images = sess.run([train_im])[0]
                        if len(images) < args.batch_size:
                            images = sess.run([train_im])[0]

                        inp_dict = {}
                        inp_dict = loss.feed_all_gpu(inp_dict, args.num_gpu, args.batch_size_per_gpu, images, params)
                        inp_dict[learning_rate] = learning_rate_val
                        inp_dict[lambda_rec] = 1.

                        _ = sess.run(
                            [train_op_G],
                            feed_dict=inp_dict)
                    print('Pre-train G Done!')

                if (iters < 25 and args.checkpoint_path is None) or iters % 500 == 0:
                    n_cir = 30
                else:
                    n_cir = args.critic_steps

                for t in range(n_cir):
                    images = sess.run([train_im])[0]
                    if len(images) < args.batch_size:
                        images = sess.run([train_im])[0]

                    inp_dict = {}
                    inp_dict = loss.feed_all_gpu(inp_dict, args.num_gpu, args.batch_size_per_gpu, images, params)
                    inp_dict[learning_rate] = learning_rate_val
                    inp_dict[lambda_rec] = args.lambda_rec

                    _ = sess.run(
                        [train_op_D],
                        feed_dict=inp_dict)


                if iters % 50 == 0:

                    _, g_val, ag_val, rs, d_val = sess.run(
                        [train_op_G, aver_loss_g, aver_loss_ag, merged, aver_loss_d],
                        feed_dict=inp_dict)
                    writer.add_summary(rs, iters)

                else:

                    _, g_val, ag_val, d_val = sess.run(
                        [train_op_G, aver_loss_g, aver_loss_ag, aver_loss_d],
                        feed_dict=inp_dict)
                if iters % 20 == 0:
                    print("Iter:", iters, 'loss_g:', g_val, 'loss_d:', d_val, 'loss_adv_g:', ag_val)
                    itimer.lap()
                iters += 1
            saver.save(sess, model_path, global_step=iters)


            # testing
            if epoch > 0:
                ii = 0
                g_vals = 0
                d_vals = 0
                ag_vals = 0
                n_batchs = 0
                for _ in range(int(args.testset_length / args.batch_size)):
                    test_oris = sess.run([test_im])[0]
                    if len(test_oris) < args.batch_size:
                        test_oris = sess.run([test_im])[0]

                    inp_dict = {}
                    inp_dict = loss.feed_all_gpu(inp_dict, args.num_gpu, args.batch_size_per_gpu, test_oris, params)
                    inp_dict[learning_rate] = learning_rate_val
                    inp_dict[lambda_rec] = args.lambda_rec

                    reconstruction_vals, g_val, d_val, ag_val = sess.run(
                        [reconstruction, aver_loss_g, aver_loss_d, aver_loss_ag],
                        feed_dict=inp_dict)

                    g_vals += g_val
                    d_vals += d_val
                    ag_vals += ag_val
                    n_batchs += 1

                    # Save test results
                    if epoch % 100 == 0:

                        for rec_val, test_ori in zip(reconstruction_vals, test_oris):
                            rec_hid = (255. * (rec_val + 1) /
                                       2.).astype(np.uint8)
                            test_ori = (255. * (test_ori + 1) /
                                       2.).astype(np.uint8)
                            Image.fromarray(rec_hid).save(os.path.join(
                                result_path, 'img_' + str(ii) + '.' + str(int(iters / 100)) + '.jpg'))
                            if epoch == 0:
                                Image.fromarray(test_ori).save(
                                    os.path.join(result_path, 'img_' + str(ii) + '.' + str(int(iters / 100)) + '.ori.jpg'))
                            ii += 1
                g_vals /= n_batchs
                d_vals /= n_batchs
                ag_vals /= n_batchs

                summary = tf.compat.v1.Summary()
                summary.value.add(tag='eval/g',
                                  simple_value=g_vals)
                summary.value.add(tag='eval/d',
                                  simple_value=d_vals)
                summary.value.add(tag='eval/ag',
                                  simple_value=ag_vals)
                writer.add_summary(summary, iters)

                print("=========================================================================")
                print('loss_g:', g_val, 'loss_d:', d_val, 'loss_adv_g:', ag_val)
                print("=========================================================================")

                if np.isnan(reconstruction_vals.min()) or np.isnan(reconstruction_vals.max()):
                    print("NaN detected!!")
            etimer.stop()

#cProfile.run('main()')