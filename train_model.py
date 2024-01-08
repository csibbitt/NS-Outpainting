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
parser.add_argument('--resume-step', type=int, default=0)

# determinism
parser.add_argument('--deterministic-seed', type=int, default=0)

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
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
args.batch_size_per_gpu = int(args.batch_size / args.num_gpu)


generator = Generator(args)
loss = Loss(args)

print("Start building model...")
with tf.compat.v1.Session() as sess:

    writer = tf.summary.create_file_writer(tensorboard_path)
    sess.run(writer.init())

    if args.deterministic_seed != 0:
        tf.compat.v1.random.set_random_seed(1)
        tf.keras.utils.set_random_seed(1)
        tf.config.experimental.enable_op_determinism()
        tf.random.set_seed(1)

    with tf.device('/cpu:0'):

        btimer = Timer('building')
        learning_rate = tf.compat.v1.placeholder(tf.float32, [])
        lambda_rec = tf.compat.v1.placeholder(tf.float32, [])

        G_opt = tf.keras.optimizers.legacy.Adam(
            learning_rate=learning_rate, beta_1=0.5, beta_2=0.9, epsilon=1e-08)
        D_opt = tf.keras.optimizers.legacy.Adam(
            learning_rate=learning_rate, beta_1=0.5, beta_2=0.9, epsilon=1e-08)


        trainset = tf.data.TFRecordDataset(filenames=[args.trainset_path])
        trainset = trainset.shuffle(args.trainset_length)
        trainset = trainset.map(parse_trainset, num_parallel_calls=args.workers)
        trainset = trainset.batch(args.batch_size).repeat()

        train_iterator = tf.compat.v1.data.make_one_shot_iterator(trainset) #**
        train_im = train_iterator.get_next()

        testset = tf.data.TFRecordDataset(filenames=[args.testset_path])
        testset = testset.map(parse_testset, num_parallel_calls=args.workers)
        testset = testset.batch(args.batch_size).repeat()

        test_iterator = tf.compat.v1.data.make_one_shot_iterator(testset) #**
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

                        loss_rec = loss.masked_reconstruction_loss(groundtruth, reconstruction)
                        loss_adv_G, loss_adv_D = loss.global_and_local_adv_loss(groundtruth, reconstruction)

                        #***** This is broken, always returns an empty collection
                        # reg_losses = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)
                        # Suspect regularization penalties are nerfed - or does TF2 apply them automatically at each layer? (doubt)
                        # Line should be this; and early visual results seem okay:
                        reg_losses = generator.losses
                        # The old code was finding 99 tensors, but the new code finds 201. Hmmm....
                        #*****

                        loss_G = loss_adv_G * (1 - lambda_rec) + loss_rec * lambda_rec + sum(reg_losses)
                        loss_D = loss_adv_D

                        var_G = generator.trainable_variables
                        var_D = loss.discrim_l.trainable_variables + loss.discrim_g.trainable_variables

                        grad_g = G_opt.get_gradients( #** When moving off legacy.Adam this will need to change back to compute_gradients
                            loss_G, var_G)
                        grad_g = zip(grad_g, var_G) #** Required because get_gradients only returns grads but compute_gradients returns grad,var tuples
                        grad_d = D_opt.get_gradients(
                            loss_D, var_D)
                        grad_d = zip(grad_d, var_D)

                        models.append((grad_g, grad_d, loss_G, loss_D, loss_adv_G, loss_rec, reconstruction))
                        params.append(groundtruth)

        print('Done.')
        btimer.stop()
        generator.summary()

        print('Start reducing towers on gpu...')

        grad_gs, grad_ds, loss_Gs, loss_Ds, loss_adv_Gs, loss_recs, reconstructions = zip(*models)
        groundtruths = params

        iters = 0
        step = tf.Variable(0, dtype=tf.int64, trainable=False)
        with tf.device('/gpu:0'):

            aver_loss_g = tf.reduce_mean(input_tensor=loss_Gs)
            aver_loss_d = tf.reduce_mean(input_tensor=loss_Ds)
            aver_loss_ag = tf.reduce_mean(input_tensor=loss_adv_Gs)
            aver_loss_rec = tf.reduce_mean(input_tensor=loss_recs)

            train_op_G = G_opt.apply_gradients(
                loss.average_gradients(grad_gs), experimental_aggregate_gradients=False) #** skip_gradients_aggregation=True when moving off legacy (unless I sort out the gradient averaging since it defaults to a sum)
            train_op_D = D_opt.apply_gradients(
                loss.average_gradients(grad_ds), experimental_aggregate_gradients=False)

            groundtruths = tf.concat(groundtruths, axis=0)
            reconstructions = tf.concat(reconstructions, axis=0)

            with writer.as_default(step=step):
                tf.summary.scalar('loss_g', aver_loss_g)
                tf.summary.scalar('loss_d', aver_loss_d)
                tf.summary.scalar('loss_ag', aver_loss_ag)
                tf.summary.scalar('loss_rec', aver_loss_rec)
                tf.summary.image('groundtruth', tf.dtypes.cast(((groundtruths + 1) * 255. / 2.), tf.uint8), max_outputs=2)
                tf.summary.image('reconstruction', tf.dtypes.cast(((reconstructions + 1) * 255. / 2.), tf.uint8), max_outputs=2)

        print('Done.')

        ckpt = tf.train.Checkpoint(step=step,
                                   G_opt=G_opt,
                                   D_opt=D_opt,
                                   generator=generator,
                                   discrim_g=loss.discrim_g,
                                   discrim_l=loss.discrim_l)
        print('Creating v2 checkpoint manager')
        ckpt_manager = tf.train.CheckpointManager(ckpt, directory=ckpt_path, max_to_keep=5)
        saver = tf.compat.v1.train.Saver(max_to_keep=3) # *****

        if args.checkpoint_path is None:
            sess.run(tf.compat.v1.global_variables_initializer())
        else:
            if args.load_v2_checkpoint:
                print('Start v2 loading checkpoint...')
                ckpt_path = args.checkpoint_path
                status = ckpt.restore(args.checkpoint_path)
                status.assert_consumed()
                iters = step
            else:
                print('Start v1 loading checkpoint...')
                saver.restore(sess, args.checkpoint_path)
                iters = args.resume_step
            print('Start loading checkpoint...')
            saver.restore(sess, args.checkpoint_path)
            iters = args.resume_step
            print('Done.')

        print('Start training...')
        for epoch in range(args.epoch):
            etimer = Timer(f'epoch {epoch}')
            if epoch > args.lr_decay_epoch:
                learning_rate_val = args.base_lr / 10  # ***** This looks like a bug, after loading a checkpoing, lr will be high for some epochs
                                                       # Would need to save/load epoch in the checkpoint
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

                    _, g_val, ag_val, d_val, _ = sess.run(
                        [train_op_G, aver_loss_g, aver_loss_ag, aver_loss_d, tf.compat.v1.summary.all_v2_summary_ops()],
                        feed_dict=inp_dict)
                    sess.run(writer.flush())
                else:

                    _, g_val, ag_val, d_val = sess.run(
                        [train_op_G, aver_loss_g, aver_loss_ag, aver_loss_d],
                        feed_dict=inp_dict)
                if iters % 20 == 0:
                    print("Iter:", iters, 'loss_g:', g_val, 'loss_d:', d_val, 'loss_adv_g:', ag_val)
                    itimer.lap()
                iters += 1
                sess.run(step.assign(iters))
            ckpt_manager.save()


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

                with writer.as_default(step=step):
                    tf.summary.scalar('eval/g', g_vals)
                    tf.summary.scalar('eval/d', d_vals)
                    tf.summary.scalar('eval/ag', ag_vals)
                sess.run(writer.flush())

                print("=========================================================================")
                print('loss_g:', g_val, 'loss_d:', d_val, 'loss_adv_g:', ag_val)
                print("=========================================================================")

                if np.isnan(reconstruction_vals.min()) or np.isnan(reconstruction_vals.max()):
                    print("NaN detected!!")
            etimer.stop()

#cProfile.run('main()')