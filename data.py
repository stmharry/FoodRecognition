'''Provides Data for Food Classification
'''

from __future__ import print_function

import numpy as np
import os
import random
import scipy.io
import tensorflow as tf

from deepbox import image_util

CACHE_NAME = 'cache'
RESNET_MAT_PATH = '/mnt/data/ResNet-50-params.mat'  # TO CHANGE
RESNET_MAT = scipy.io.loadmat(RESNET_MAT_PATH)

FILENAME_CLASSNAME_LIST = []
CLASS_NAMES = []
SIZE = 224
DEV_VAL_SPLIT = 63

TRAIN = 0
DEV = 1
VAL = 2
TEST = 3
NAMES = {TRAIN: 'train', DEV: 'dev', VAL: 'val', TEST: 'test'}


def cache_train_files(train_dir, recache=False):
    # FILENAME_CLASSNAME_LIST = [(filename0, classname0), (filename1, classname1), ...]
    global FILENAME_CLASSNAME_LIST, CLASS_NAMES
    if FILENAME_CLASSNAME_LIST and CLASS_NAMES:
        return

    cache_path = os.path.join(train_dir, CACHE_NAME)
    if os.path.isfile(cache_path) and not recache:
        FILENAME_CLASSNAME_LIST = np.loadtxt(cache_path, dtype=np.str, delimiter=',', comments=None)
    else:
        FILENAME_CLASSNAME_LIST = []
        for class_name in os.listdir(train_dir):
            class_dir = os.path.join(train_dir, class_name)
            # Reject non-`class_name` directories
            if class_name.startswith('.') or not os.path.isdir(class_dir):
                continue
            # Walk though each file in `train_dir/class_name`
            for (file_dir, _, file_names) in os.walk(class_dir):
                for file_name in file_names:
                    # Get rid of some hidden metadata
                    if not file_name.endswith('.jpg'):
                        continue
                    FILENAME_CLASSNAME_LIST.append((os.path.join(file_dir, file_name), class_name))
        np.savetxt(cache_path, FILENAME_CLASSNAME_LIST, fmt='%s', delimiter=',')
    CLASS_NAMES = np.unique(zip(*FILENAME_CLASSNAME_LIST)[1])


def get_files(phase, num_pipelines, test_dir=None):
    filename_label_list_list = [[] for _ in xrange(num_pipelines)]
    if phase == TRAIN or phase == DEV or phase == VAL:
        for (file_name, class_name) in FILENAME_CLASSNAME_LIST:
            filename_hash = hash(file_name)
            label = np.argwhere(CLASS_NAMES == class_name)[0, 0]
            if phase == TRAIN or (phase == DEV and filename_hash % DEV_VAL_SPLIT != 0) or (phase == VAL and filename_hash % DEV_VAL_SPLIT == 0):
                filename_label_list_list[filename_hash % num_pipelines].append((file_name, label))
    elif phase == TEST:
        for (file_dir, _, file_names) in os.walk(test_dir):
            for file_name in file_names:
                if not file_name.endswith('.jpg'):
                    continue
                filename_hash = hash(file_name)
                filename_label_list_list[filename_hash % num_pipelines].append((os.path.join(file_dir, file_name), -1))
    print('File phase %s, number of files=%d, number of pipelines=%d' % (NAMES[phase], sum(map(len, filename_label_list_list)), num_pipelines))
    return filename_label_list_list


def _get_queue(value_list, dtype):
    queue = tf.FIFOQueue(32, dtypes=[dtype], shapes=[()])
    enqueue = queue.enqueue_many([value_list])
    queue_runner = tf.train.QueueRunner(queue, [enqueue])
    tf.train.add_queue_runner(queue_runner)
    return queue


def _get_values(filename_label_list, shuffle):
    if shuffle:
        random.shuffle(filename_label_list)

    (filename_list, label_list) = map(list, zip(*filename_label_list))

    filename_queue = _get_queue(filename_list, tf.string)
    label_queue = _get_queue(label_list, tf.int32)

    reader = tf.WholeFileReader()
    (key, value) = reader.read(filename_queue)
    image = tf.image.decode_jpeg(value)
    image = tf.to_float(image)

    label = label_queue.dequeue()
    label = tf.to_int64(label)

    return (key, image, label)


def get_train_values(filename_label_list_list, batch_size):
    # A list of `(key, image, label)` from various workers
    values_list = []
    for filename_label_list in filename_label_list_list:
        (key, image, label) = _get_values(filename_label_list, shuffle=True)
        image = image_util.random_resize(image, size_range=(256, 512))
        image = image_util.random_crop(image, size=SIZE)
        image = image_util.random_flip(image)
        image = image_util.random_adjust_rgb(image)
        image = image - RESNET_MAT['mean']

        values_list.append((key, image, label))

    (key, image, label) = tf.train.shuffle_batch_join(
        values_list,
        batch_size=batch_size,
        capacity=4096 + 512,
        min_after_dequeue=4096)

    return (key, image, label)


def get_test_values(filename_label_list_list, batch_size, num_test_crops):
    values_list = []
    for filename_label_list in filename_label_list_list:
        (key, image, label) = _get_values(filename_label_list, shuffle=False)
        image = tf.tile(tf.expand_dims(image, dim=0), multiples=(num_test_crops, 1, 1, 1))

        def test_image_map(image):
            image = image_util.random_resize(image, size_range=(384, 384))
            image = image_util.random_crop(image, size=SIZE)
            image = image_util.random_flip(image)
            image = image - RESNET_MAT['mean']
            return image

        image = tf.map_fn(test_image_map, image)
        image.set_shape((num_test_crops, SIZE, SIZE, 3))

        values_list.append((key, image, label))

    test_batch_size = int(batch_size / num_test_crops)
    (key, image, label) = tf.train.batch_join(
        values_list,
        batch_size=test_batch_size,
        capacity=test_batch_size)
    image = tf.reshape(image, shape=(-1, SIZE, SIZE, 3))

    return (key, image, label)
