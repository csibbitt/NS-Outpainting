import random
from glob import glob
import numpy as np
from PIL import Image
import tensorflow as tf
from dataset.build_dataset import input_hasher
from dataset.parse import parse_testset

testset_path = './dataset/testset.tfr'

def mainSession(buffer_size, img_callback, shuffle_flag, input_images, seed_hash, mix_strength):

    print("Start building model...")
    with tf.device('/cpu:0'):

        testset = tf.data.TFRecordDataset(filenames=[testset_path])
        testset = testset.map(parse_testset)
        testset = testset.batch(1).repeat()
        testset_iter = iter(testset)

        generator = tf.keras.models.load_model("saved-model.tf")

        print('run eval...')
        stitch_mask1 = np.ones((1, 128, 128, 3))
        for i in range(128):
            stitch_mask1[:, :, i, :] = 1. / 127. * (127. - i)
        stitch_mask2 = stitch_mask1[:, :, ::-1, :]

        while True:
            if len(input_images) == 0:
                test_oris = testset_iter.get_next()
                seed_hash.set('t_' + input_hasher(test_oris))
            else:
                # Combo from build_dataset and train_model
                image_bytes =  input_images[0].tobytes()
                features = {'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))}
                tf_features = tf.train.Features(feature=features)
                tf_example = tf.train.Example(features=tf_features)
                input_image_tensor = parse_testset(tf_example.SerializeToString(), shape=[1, 128, 128, 3])
                input_image_tensor = tf.concat((input_image_tensor, tf.zeros([1, 128, 128, 3])), axis=2)
                test_oris = input_image_tensor
            origins1 = tf.identity(test_oris)

            oris = None
            while not shuffle_flag.get():
                with tf.device('/gpu:0'):
                    reconstruction_vals = generator(tf.slice(test_oris, [0, 0, 0, 0], [1, 128, 128, 3]))
                prediction_vals = tf.slice(reconstruction_vals, [0, 0, 128, 0], [1, 128, 128, 3])

                if oris is None:
                    oris = reconstruction_vals
                    pred1 = oris[:, :, :128, :]
                    pred2 = oris[:, :, -128:, :]
                    gt = origins1[:, :, :128, :]
                    p1_m1 = np.concatenate((gt * stitch_mask1 + pred1 * stitch_mask2, pred2), axis=2)
                    img_callback(Image.fromarray((255. * (p1_m1[0] + 1) / 2.).astype(np.uint8)))
                    prediction_count = 2
                else:
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