import tensorflow as tf
import numpy as np
import math

from model.discriminator import DiscriminatorLocal, DiscriminatorGlobal

class Loss():
    def __init__(self, cfg):
        self.cfg = cfg
        self.discrim_g = DiscriminatorGlobal(name='DIS')
        self.discrim_l = DiscriminatorLocal(name='DIS2')

    def masked_reconstruction_loss(self, gt, recon):
        loss_recon = tf.square(gt - recon)
        mask_values = np.ones((128, 128))
        for j in range(128):
            mask_values[:, j] = (1. + math.cos(math.pi * j / 127.0)) * 0.5
        mask_values = np.expand_dims(mask_values, 0)
        mask_values = np.expand_dims(mask_values, 3)
        mask1 = tf.constant(1, dtype=tf.float32, shape=[1, 128, 128, 1])
        mask2 = tf.constant(mask_values, dtype=tf.float32, shape=[1, 128, 128, 1])
        mask = tf.concat([mask1, mask2], axis=2)
        loss_recon = loss_recon * mask
        loss_recon = tf.reduce_mean(input_tensor=loss_recon)
        return loss_recon

    def adversarial_loss(self, dis_fun, real, fake):
        adversarial_pos = dis_fun(real)
        adversarial_neg = dis_fun(fake)

        loss_adv_D = - tf.reduce_mean(input_tensor=adversarial_pos - adversarial_neg)

        differences = fake - real
        alpha = tf.random.uniform(shape=[self.cfg.batch_size_per_gpu, 1, 1, 1])
        interpolates = real + tf.multiply(alpha, differences)
        with tf.GradientTape() as g:
            g.watch(interpolates)
            ys = dis_fun(interpolates)

        gradients = g.gradient(ys, interpolates)
        slopes = tf.sqrt(tf.reduce_sum(
            input_tensor=tf.square(gradients), axis=[1, 2, 3]) + 1e-10)
        gradients_penalty = tf.reduce_mean(input_tensor=(slopes - 1.) ** 2)
        loss_adv_D += self.cfg.lambda_gp * gradients_penalty

        loss_adv_G = -tf.reduce_mean(input_tensor=adversarial_neg)

        return loss_adv_D, loss_adv_G

    def global_and_local_adv_loss(self, gt, recon):

        left_half_gt = tf.slice(gt, [0, 0, 0, 0], [self.cfg.batch_size_per_gpu, 128, 128, 3])
        right_half_gt = tf.slice(gt, [0, 0, 128, 0], [self.cfg.batch_size_per_gpu, 128, 128, 3])
        right_half_recon = tf.slice(recon, [0, 0, 128, 0], [self.cfg.batch_size_per_gpu, 128, 128, 3])
        real = gt
        fake = tf.concat([left_half_gt, right_half_recon], axis=2)
        global_D, global_G = self.adversarial_loss(self.discrim_g, real, fake)

        real = right_half_gt
        fake = right_half_recon
        local_D, local_G = self.adversarial_loss(self.discrim_l, real, fake)

        loss_adv_D = global_D + local_D
        loss_adv_G = self.cfg.beta * global_G + (1 - self.cfg.beta) * local_G

        return loss_adv_G, loss_adv_D

    # def average_gradients(self, tower_grads):
    #     average_grads = []
    #     for grad_and_vars in zip(*tower_grads):
    #         # Note that each grad_and_vars looks like the following:
    #         #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    #         grads = []
    #         # Average over the 'tower' dimension.
    #         g, _ = grad_and_vars[0]

    #         for g, _ in grad_and_vars:
    #             expanded_g = tf.expand_dims(g, 0)
    #             grads.append(expanded_g)
    #         grad = tf.concat(grads, axis=0)
    #         grad = tf.reduce_mean(input_tensor=grad, axis=0)

    #         # Keep in mind that the Variables are redundant because they are shared
    #         # across towers. So .. we will just return the first tower's pointer to
    #         # the Variable.
    #         v = grad_and_vars[0][1]
    #         grad_and_var = (grad, v)
    #         average_grads.append(grad_and_var)
    #     # clip
    #     if self.cfg.clip_gradient:
    #         gradients, variables = zip(*average_grads)
    #         gradients = [
    #             None if gradient is None else tf.compat.v1.clip_by_average_norm(gradient, self.cfg.clip_gradient_value)
    #             for gradient in gradients]
    #         average_grads = zip(gradients, variables)
    #     return average_grads

    # def feed_all_gpu(self, inp_dict, gpu_num, payload_per_gpu, images, params):
    #     for i in range(gpu_num):
    #         gt = params[i]
    #         start_pos = i * payload_per_gpu
    #         stop_pos = (i + 1) * payload_per_gpu
    #         inp_dict[gt] = images[start_pos:stop_pos]
    #     return inp_dict


