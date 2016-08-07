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

FILENAME_LABEL_LIST = []
HASH_LIST = []
CLASS_NAMES = []
SIZE = 224
DEV_VAL_SPLIT = 63

TRAIN = 0
DEV = 1
VAL = 2
TEST = 3
NAMES = {TRAIN: 'train', DEV: 'dev', VAL: 'val', TEST: 'test'}


def cache_train_files(directory, recache=False):
    # FILENAME_LABEL_LIST = [(filename0, label0), (filename1, label1), ...]
    global FILENAME_LABEL_LIST, HASH_LIST, CLASS_NAMES

    cache_path = os.path.join(directory, CACHE_NAME)
    if os.path.isfile(cache_path) and not recache:
        filename_list, classname_list = np.loadtxt(cache_path, dtype=np.str, delimiter=',', comments=None, unpack=True)
    else:
        filename_list = []
        classname_list = []
        for class_name in os.listdir(directory):
            class_dir = os.path.join(directory, class_name)
            # Reject non-`class_name` directories
            if class_name.startswith('.') or not os.path.isdir(class_dir):
                continue
            # Walk though each file in `directory/class_name`
            for (file_dir, _, file_names) in os.walk(class_dir):
                for file_name in file_names:
                    # Get rid of some hidden metadata
                    if not file_name.endswith('.jpg'):
                        continue
                    filename_list.append(os.path.join(file_dir, file_name))
                    classname_list.append(class_name)
        np.savetxt(cache_path, zip(filename_list, classname_list), fmt='%s', delimiter=',')

    CLASS_NAMES = sorted(set(classname_list))
    FILENAME_LABEL_LIST = zip(filename_list, map(CLASS_NAMES.index, classname_list))
    HASH_LIST = map(hash, filename_list)


def get_files(phase, num_pipelines=1, subsample_ratio=1.0, directory=None):
    if phase == TRAIN or phase == DEV or phase == VAL:
        if directory is not None and not CLASS_NAMES:
            cache_train_files(directory)

    filename_label_list_list = [[] for _ in xrange(num_pipelines)]

    if phase == TRAIN:
        filename_label_list = FILENAME_LABEL_LIST
        hash_list = HASH_LIST
    elif phase == DEV:
        sel = [i for (i, hash_) in enumerate(HASH_LIST) if hash_ % DEV_VAL_SPLIT != 0]
        filename_label_list = map(FILENAME_LABEL_LIST.__getitem__, sel)
        hash_list = map(HASH_LIST.__getitem__, sel)
    elif phase == VAL:
        sel = [i for (i, hash_) in enumerate(HASH_LIST) if hash_ % DEV_VAL_SPLIT == 0]
        filename_label_list = map(FILENAME_LABEL_LIST.__getitem__, sel)
        hash_list = map(HASH_LIST.__getitem__, sel)
    elif phase == TEST:
        filename_label_list = []
        for (file_dir, _, file_names) in os.walk(directory):
            for file_name in file_names:
                if file_name.endswith('.jpg'):
                    filename_label_list.append((os.path.join(file_dir, file_name), -1))
        hash_list = map(hash, zip(*filename_label_list)[0])

    if subsample_ratio != 1.0:
        print('[ Class-wise subsample ]')
        filename_label_list_ = []
        for (idx_class, class_name) in enumerate(CLASS_NAMES):
            sel_class = [i for (i, (filename, label)) in enumerate(filename_label_list) if label == idx_class]
            num_samples = len(sel_class)
            num_subsamples = int(subsample_ratio * num_samples)
            filename_label_list_ += map(filename_label_list.__getitem__, random.sample(sel_class, num_subsamples))
            print('%s: %d -> %d' % (class_name, num_samples, num_subsamples))
        filename_label_list = filename_label_list_

    for (filename_label, hash_) in zip(filename_label_list, hash_list):
        filename_label_list_list[hash_ % num_pipelines].append(filename_label)

    print('File phase %s, number of files=%d, number of pipelines=%d' % (NAMES[phase], len(filename_label_list), num_pipelines))
    return filename_label_list_list


def _get_queue(value_list, dtype):
    queue = tf.FIFOQueue(32, dtypes=[dtype], shapes=[()])
    enqueue = queue.enqueue_many([value_list])
    queue_runner = tf.train.QueueRunner(queue, [enqueue])
    tf.train.add_queue_runner(queue_runner)
    return queue


def _get_values(filename_queue, label_queue):
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
        random.shuffle(filename_label_list)
        (filename_list, label_list) = map(list, zip(*filename_label_list))
        filename_queue = _get_queue(filename_list, tf.string)
        label_queue = _get_queue(label_list, tf.int32)

        (key, image, label) = _get_values(filename_queue, label_queue)
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
        (filename_list, label_list) = map(list, zip(*filename_label_list))
        filename_queue = _get_queue(filename_list, tf.string)
        label_queue = _get_queue(label_list, tf.int32)

        (key, image, label) = _get_values(filename_queue, label_queue)
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
