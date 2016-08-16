'''
TODO
1. Image reading threads
'''

from __future__ import print_function

import enum
import numpy as np
import os
import scipy.io
import subprocess
import sys
import tensorflow as tf
import time

from deepbox import util, image_util
from deepbox.model import Model


class Meta(object):
    WORKING_DIR = '/tmp/' + time.strftime('%Y%-m-%d-%H%M%S')
    CLASSNAMES_FILENAME = 'class_names.txt'
    CLASS_NAMES = list()

    @staticmethod
    def classnamesPath():
        return os.path.join(Meta.WORKING_DIR, Meta.CLASSNAMES_FILENAME)

    @staticmethod
    def save(class_names=CLASS_NAMES):
        Meta.CLASS_NAMES = class_names

        if not os.path.isdir(Meta.WORKING_DIR):
            os.makedirs(Meta.WORKING_DIR)
        np.savetxt(Meta.classnamesPath(), Meta.CLASS_NAMES, delimiter=',', fmt='%s')

    @staticmethod
    def load(working_dir=WORKING_DIR):
        Meta.WORKING_DIR = working_dir

        if os.path.isfile(Meta.classnamesPath()):
            Meta.CLASS_NAMES = np.loadtxt(Meta.classnamesPath(), dtype=np.str, delimiter=',')


class Blob(object):
    def __init__(self, image=None, label=None, images=None, labels=None, imageLabels=None):
        assert (image is not None) + (images is not None) + (imageLabels is not None) == 1, 'Too many arguments!'

        if image is not None:
            label = tf.constant(-1, dtype=tf.int64) if label is None else label
            self.image = image
            self.label = label

            images = [image]
            labels = [label]
        elif images is not None:
            labels = [tf.constant(-1, dtype=tf.int64) for _ in xrange(len(images))] if labels is None else labels
        else:
            (images, labels) = zip(*imageLabels)

        self.images = images
        self.labels = labels

    def as_tuple_list(self):
        return zip(self.images, self.labels)

    def func(self, f):
        return f(self)


class Producer(object):
    NUM_TRAIN_INPUTS = 8
    NUM_TEST_INPUTS = 1
    SUBSAMPLE_SIZE = 64

    @staticmethod
    def get_queue(value_list, dtype):
        queue = tf.FIFOQueue(32, dtypes=[dtype], shapes=[()])
        enqueue = queue.enqueue_many([value_list])
        queue_runner = tf.train.QueueRunner(queue, [enqueue])
        tf.train.add_queue_runner(queue_runner)
        return queue

    def __init__(self,
                 num_train_inputs=NUM_TRAIN_INPUTS,
                 num_test_inputs=NUM_TEST_INPUTS,
                 subsample_size=SUBSAMPLE_SIZE):

        self.num_train_inputs = num_train_inputs
        self.num_test_inputs = num_test_inputs
        self.subsample_size = subsample_size

    def _file(self,
              image_dir,
              num_inputs=1,
              subsample_size=1,
              subsample_divisible=True,
              is_train=False,
              check=False,
              shuffle=False):

        filename_list = list()
        classname_list = list()

        for class_name in os.listdir(image_dir):
            class_dir = os.path.join(image_dir, class_name)
            if class_name.startswith('.') or not os.path.isdir(class_dir):
                continue
            for (file_dir, _, file_names) in os.walk(class_dir):
                for file_name in file_names:
                    if not file_name.endswith('.jpg'):
                        continue
                    if (hash(file_name) % subsample_size == 0) != subsample_divisible:
                        continue
                    filename_list.append(os.path.join(file_dir, file_name))
                    classname_list.append(class_name)

        class_names = sorted(set(classname_list))
        label_list = map(class_names.index, classname_list)

        if is_train:
            Meta.save(class_names=class_names)

        if check:
            num_file_list = list()
            for (num_file, filename) in enumerate(filename_list):
                print('\033[2K\rChecking image %d / %d' % (num_file + 1, len(filename_list)), end='')
                sp = subprocess.Popen(['identify', filename], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                (stdout, stderr) = sp.communicate()
                if stderr:
                    os.remove(filename)
                    print('\nRemove %s' % filename)
                else:
                    num_file_list.append(num_file)
                sys.stdout.flush()
            print('')

            filename_list = map(filename_list.__getitem__, num_file_list)
            label_list = map(label_list.__getitem__, num_file_list)

        imageLabels = list()
        for num_input in xrange(num_inputs):
            if shuffle:
                perm = np.random.permutation(len(filename_list))
                filename_list = map(filename_list.__getitem__, perm)
                label_list = map(label_list.__getitem__, perm)

            filename_queue = Producer.get_queue(filename_list, dtype=tf.string)
            (key, value) = tf.WholeFileReader().read(filename_queue)
            image = tf.to_float(tf.image.decode_jpeg(value))

            label_queue = Producer.get_queue(label_list, dtype=tf.int64)
            label = label_queue.dequeue()

            imageLabels.append((image, label))

        return Blob(imageLabels=imageLabels)

    def trainFile(self,
                  image_dir,
                  subsample_divisible=False,
                  is_train=True,
                  check=True,
                  shuffle=True):

        return self._file(image_dir,
                          num_inputs=self.num_train_inputs,
                          subsample_size=self.subsample_size,
                          subsample_divisible=subsample_divisible,
                          is_train=is_train,
                          check=check,
                          shuffle=shuffle)

    def testFile(self,
                 image_dir,
                 subsample_divisible=True,
                 is_train=False,
                 check=False,
                 shuffle=False):

        return self._file(image_dir,
                          num_inputs=self.num_test_inputs,
                          subsample_size=self.subsample_size,
                          subsample_divisible=subsample_divisible,
                          is_train=is_train,
                          check=check,
                          shuffle=shuffle)


class Preprocess(object):
    NUM_TEST_CROPS = 4
    TRAIN_SIZE_RANGE = (256, 512)
    TEST_SIZE_RANGE = (384, 384)
    NET_SIZE = 224
    NET_CHANNEL = 3
    MEAN_PATH = 'archive/ResNet-mean.mat'

    def __init__(self,
                 num_test_crops=NUM_TEST_CROPS,
                 train_size_range=TRAIN_SIZE_RANGE,
                 test_size_range=TEST_SIZE_RANGE,
                 net_size=NET_SIZE,
                 net_channel=NET_CHANNEL,
                 mean_path=MEAN_PATH):

        self.num_test_crops = num_test_crops
        self.train_size_range = train_size_range
        self.test_size_range = test_size_range

        self.net_size = net_size
        self.net_channel = net_channel
        self.shape = (net_size, net_size, net_channel)

        self.mean_path = mean_path
        self.mean = scipy.io.loadmat(mean_path)['mean']

    def _train(self, image):
        image = image_util.random_resize(image, size_range=self.train_size_range)
        image = image_util.random_crop(image, size=self.net_size)
        image = image_util.random_flip(image)
        image = image_util.random_adjust_rgb(image)
        image = image - self.mean
        image.set_shape(self.shape)

        return image

    def train(self, blob):
        return Blob(images=map(self._train, blob.images), labels=blob.labels)

    def _test_map(self, image):
        image = image_util.random_resize(image, size_range=self.test_size_range)
        image = image_util.random_crop(image, size=self.net_size)
        image = image_util.random_flip(image)
        image = image - self.mean

        return image

    def _test(self, image):
        image = tf.tile(tf.expand_dims(image, dim=0), multiples=(self.num_test_crops, 1, 1, 1))
        image = tf.map_fn(self._test_map, image)
        image.set_shape((self.num_test_crops,) + self.shape)

        return image

    def test(self, blob):
        return Blob(images=map(self._test, blob.images), labels=blob.labels)


class Batch(object):
    BATCH_SIZE = 64
    CAPACITY = 4096 + 1024
    MIN_AFTER_DEQUEUE = 4096
    NUM_TEST_CROPS = 4

    def __init__(self,
                 batch_size=BATCH_SIZE,
                 capacity=CAPACITY,
                 min_after_dequeue=MIN_AFTER_DEQUEUE,
                 num_test_crops=NUM_TEST_CROPS):

        self.batch_size = batch_size
        self.capacity = capacity
        self.min_after_dequeue = min_after_dequeue
        self.num_test_crops = num_test_crops

    def train(self, blob):
        (image, label) = tf.train.shuffle_batch_join(
            blob.as_tuple_list(),
            batch_size=self.batch_size,
            capacity=self.capacity,
            min_after_dequeue=self.min_after_dequeue)
        return Blob(image=image, label=label)

    def test(self, blob):
        (image, label) = tf.train.batch_join(
            blob.as_tuple_list(),
            batch_size=self.batch_size / self.num_test_crops,
            capacity=self.batch_size / self.num_test_crops)

        shape = image_util.get_shape(image)
        image = tf.reshape(image, (self.batch_size,) + shape[2:])
        return Blob(image=image, label=label)


class Net(object):
    class Phase(enum.Enum):
        TRAIN = 0
        TEST = 1

    NET_VARIABLES = 'net_variables'
    NET_COLLECTIONS = [tf.GraphKeys.VARIABLES, NET_VARIABLES]

    LEARNING_RATE = 1e-1
    LEARNING_RATE_MODES = dict(normal=1.0, slow=0.0)
    LEARNING_RATE_DECAY_STEPS = 0
    LEARNING_RATE_DECAY_RATE = 1.0
    WEIGHT_DECAY = 0.0

    @staticmethod
    def placeholder(name, shape=None, dtype=tf.float32):
        return tf.placeholder(
            name=name,
            shape=shape,
            dtype=dtype)

    @staticmethod
    def get_const_variable(value, name, shape=(), dtype=tf.float32, trainable=False, collections=NET_COLLECTIONS):
        return tf.get_variable(
            name,
            shape=shape,
            dtype=dtype,
            initializer=tf.constant_initializer(value),
            trainable=trainable,
            collections=collections)

    @staticmethod
    def expand(size):
        return (1,) + size + (1,)

    @staticmethod
    def avg_pool(value, name, size, stride=None, padding='SAME'):
        with tf.variable_scope(name):
            if stride is None:
                stride = size
            value = tf.nn.avg_pool(value, ksize=Net.expand(size), strides=Net.expand(stride), padding=padding, name='avg_pool')
        return value

    @staticmethod
    def max_pool(value, name, size, stride=None, padding='SAME'):
        with tf.variable_scope(name):
            if stride is None:
                stride = size
            value = tf.nn.max_pool(value, ksize=Net.expand(size), strides=Net.expand(stride), padding=padding, name='max_pool')
        return value

    def __init__(self,
                 learning_rate=LEARNING_RATE,
                 learning_modes=LEARNING_RATE_MODES,
                 learning_rate_decay_steps=LEARNING_RATE_DECAY_STEPS,
                 learning_rate_decay_rate=LEARNING_RATE_DECAY_RATE,
                 weight_decay=WEIGHT_DECAY,
                 is_train=False,
                 is_show=False):
        assert len(Meta.CLASS_NAMES), 'Only create net when Meta.CLASS_NAMES is not empty!'

        self.learning_rate = Net.get_const_variable(learning_rate, 'learning_rate')
        self.learning_modes = learning_modes
        self.weight_decay = weight_decay
        self.is_train = is_train
        self.is_show = is_show

        self.phase = Net.placeholder('phase', shape=())
        self.class_names = Net.get_const_variable(Meta.CLASS_NAMES, 'class_names', shape=(len(Meta.CLASS_NAMES),), dtype=tf.string)
        self.global_step = Net.get_const_variable(0, 'global_step')
        self.checkpoint = tf.train.get_checkpoint_state(Meta.WORKING_DIR)

        if (learning_rate_decay_steps > 0) and (learning_rate_decay_rate < 1.0):
            self.learning_rate = tf.train.exponential_decay(
                self.learning_rate,
                global_step=self.global_step,
                decay_steps=learning_rate_decay_steps,
                decay_rate=learning_rate_decay_rate,
                staircase=True)

    def case(self, phase_fn_pairs, shape=None):
        pred_fn_pairs = [(tf.equal(self.phase, phase_.value), fn) for (phase_, fn) in phase_fn_pairs]
        value = tf.case(pred_fn_pairs, default=pred_fn_pairs[0][1])
        value.set_shape(shape)
        return value

    def make_stat(self):
        assert hasattr(self, 'prob'), 'net has no attribute "prob"!'

        self.target = tf.one_hot(self.label, len(Meta.CLASS_NAMES))
        self.target_frac = tf.reduce_mean(self.target, 0)
        self.loss = - tf.reduce_mean(self.target * tf.log(self.prob + util.EPSILON)) * len(Meta.CLASS_NAMES)
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        if regularization_losses:
            self.loss += tf.add_n(regularization_losses)

        self.pred = tf.argmax(self.prob, 1)
        self.correct = tf.to_float(tf.equal(self.label, self.pred))
        self.correct_frac = tf.reduce_mean(tf.expand_dims(self.correct, 1) * self.target, 0)
        self.acc = tf.reduce_mean(self.correct)

    def make_train_op(self):
        train_ops = []
        for (learning_mode, learning_rate_relative) in Net.LEARNING_RATE_MODES.iteritems():
            variables = tf.get_collection(learning_mode)
            if variables:
                train_ops.append(tf.train.AdamOptimizer(
                    learning_rate=self.learning_rate * learning_rate_relative,
                    epsilon=1.0).minimize(self.loss, global_step=self.global_step))
        self.train_op = tf.group(*train_ops)

    def make_show(self):
        def identity(value):
            return value

        postfix_funcs = {
            Net.Phase.TRAIN: {
                'raw': identity,
                'avg': lambda value: util.exponential_moving_average(value, num_updates=self.global_step)},
            Net.Phase.TEST: {
                'raw': identity,
                'avg': lambda value: util.exponential_moving_average(value, num_updates=self.global_step)}}

        self.show_dict = {
            phase: {
                '%s_%s_%s' % (phase.name, attr, postfix): func(getattr(self, attr))
                for (postfix, func) in postfix_funcs[phase].iteritems()
                for attr in ['loss', 'acc']}
            for phase in Net.Phase}

        self.show_dict[Net.Phase.TRAIN].update({
            attr: getattr(self, attr) for attr in ['learning_rate']})

        self.summary = {
            phase: tf.merge_summary([tf.scalar_summary(name, attr) for (name, attr) in self.show_dict[phase].iteritems()])
            for phase in Net.Phase}

    def finalize(self):
        self.sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
        self.saver = tf.train.Saver(tf.get_collection(Net.NET_VARIABLES), keep_checkpoint_every_n_hours=1.0)
        self.summary_writer = tf.train.SummaryWriter(Meta.WORKING_DIR)

        self.sess.run(tf.initialize_all_variables())
        if self.checkpoint:
            print('Model restored from %s' % self.checkpoint.model_checkpoint_path)
            self.saver.restore(tf.get_default_session(), self.checkpoint.model_checkpoint_path)

        tf.train.start_queue_runners()
        print('Filling queues with images...')
        self.model = Model(self.global_step)


class ResNet(Net):
    RESNET_PARAMS_PATH = 'archive/ResNet-50-params.mat'
    NUM_TEST_CROPS = 4

    def __init__(self,
                 learning_rate=Net.LEARNING_RATE,
                 learning_modes=Net.LEARNING_RATE_MODES,
                 learning_rate_decay_steps=Net.LEARNING_RATE_DECAY_STEPS,
                 learning_rate_decay_rate=Net.LEARNING_RATE_DECAY_RATE,
                 weight_decay=Net.WEIGHT_DECAY,
                 resnet_params_path=RESNET_PARAMS_PATH,
                 num_test_crops=NUM_TEST_CROPS,
                 is_train=False,
                 is_show=False):

        super(ResNet, self).__init__(
            learning_rate=learning_rate,
            learning_modes=learning_modes,
            learning_rate_decay_steps=learning_rate_decay_steps,
            learning_rate_decay_rate=learning_rate_decay_rate,
            weight_decay=weight_decay,
            is_train=is_train,
            is_show=is_show)

        self.resnet_params_path = resnet_params_path
        self.num_test_crops = num_test_crops
        if not self.checkpoint:
            self.resnet_params = scipy.io.loadmat(resnet_params_path)

    def get_initializer(self, name, index, is_vector, default):
        if self.checkpoint:
            return None
        elif name in self.resnet_params:
            print('%s initialized from ResNet' % name)
            if is_vector:
                value = self.resnet_params[name][index][0][:, 0]
            else:
                value = self.resnet_params[name][index][0]
            return tf.constant_initializer(value)
        else:
            return default

    def conv(self, value, conv_name, out_channel, size=(1, 1), stride=(1, 1), padding='SAME', biased=False, norm_name=None, activation_fn=None, learning_mode='normal'):
        in_channel = image_util.get_channel(value)

        if self.learning_modes[learning_mode] > 0:
            collections = Net.NET_COLLECTIONS + [learning_mode]
            trainable = True
        else:
            collections = Net.NET_COLLECTIONS
            trainable = False

        weights_initializer = self.get_initializer(
            conv_name,
            index=0,
            is_vector=False,
            default=tf.truncated_normal_initializer(stddev=(2. / (in_channel * stride[0] * stride[1])) ** 0.5))

        if self.weight_decay > 0:
            weight_regularizer = tf.contrib.layers.l2_regularizer(self.weight_decay)
        else:
            weight_regularizer = None

        with tf.variable_scope(conv_name):
            weight = tf.get_variable(
                'weight',
                shape=size + (in_channel, out_channel),
                initializer=weights_initializer,
                regularizer=weight_regularizer,
                trainable=trainable,
                collections=collections)

        value = tf.nn.conv2d(value, weight, strides=Net.expand(stride), padding=padding)

        if biased:
            bias_initializer = self.get_initializer(
                conv_name,
                index=1,
                is_vector=True,
                default=tf.constant_initializer(0.1))

            with tf.variable_scope(conv_name):
                bias = tf.get_variable(
                    'bias',
                    shape=(out_channel,),
                    initializer=bias_initializer,
                    trainable=trainable,
                    collections=collections)

            value = tf.nn.bias_add(value, bias)

        if norm_name is not None:
            bn_name = 'bn%s' % norm_name
            scale_name = 'scale%s' % norm_name

            mean_initializer = self.get_initializer(
                bn_name,
                index=0,
                is_vector=True,
                default=tf.constant_initializer(0.0))

            variance_initializer = self.get_initializer(
                bn_name,
                index=1,
                is_vector=True,
                default=tf.constant_initializer(1.0))

            with tf.variable_scope(bn_name):
                mean = tf.get_variable(
                    'mean',
                    shape=(out_channel,),
                    initializer=mean_initializer,
                    trainable=False,
                    collections=Net.NET_COLLECTIONS)
                variance = tf.get_variable(
                    'variance',
                    shape=(out_channel,),
                    initializer=variance_initializer,
                    trainable=False,
                    collections=Net.NET_COLLECTIONS)

            scale_initializer = self.get_initializer(
                scale_name,
                index=0,
                is_vector=True,
                default=tf.constant_initializer(1.0))

            offset_initializer = self.get_initializer(
                scale_name,
                index=1,
                is_vector=True,
                default=tf.constant_initializer(0.0))

            with tf.variable_scope(scale_name):
                scale = tf.get_variable(
                    'scale',
                    shape=(out_channel,),
                    initializer=scale_initializer,
                    trainable=trainable,
                    collections=collections)
                offset = tf.get_variable(
                    'offset',
                    shape=(out_channel,),
                    initializer=offset_initializer,
                    trainable=trainable,
                    collections=collections)

            value = (value - mean) * tf.rsqrt(variance + util.EPSILON) * scale + offset

        if activation_fn is not None:
            with tf.variable_scope(conv_name):
                value = activation_fn(value)

        print('Layer %s, shape=%s, size=%s, stride=%s, learning_mode=%s' % (value.name, value.get_shape(), size, stride, learning_mode))
        return value

    def unit(self, value, name, subsample, out_channel, learning_mode='normal'):
        in_channel = image_util.get_channel(value)

        if subsample:
            stride = (2, 2)
        else:
            stride = (1, 1)

        out_channel_inner = out_channel
        out_channel_outer = 4 * out_channel

        with tf.variable_scope(name):
            if subsample or in_channel != out_channel_outer:
                value1 = self.conv(value, 'res%s_branch1' % name, out_channel=out_channel_outer, stride=stride, norm_name='%s_branch1' % name, learning_mode=learning_mode)
            else:
                value1 = value

            value2 = self.conv(value, 'res%s_branch2a' % name, out_channel=out_channel_inner, stride=stride, norm_name='%s_branch2a' % name, activation_fn=tf.nn.relu, learning_mode=learning_mode)
            value2 = self.conv(value2, 'res%s_branch2b' % name, out_channel=out_channel_inner, size=(3, 3), norm_name='%s_branch2b' % name, activation_fn=tf.nn.relu, learning_mode=learning_mode)
            value2 = self.conv(value2, 'res%s_branch2c' % name, out_channel=out_channel_outer, norm_name='%s_branch2c' % name, learning_mode=learning_mode)

            value = tf.nn.relu(value1 + value2)
        return value

    def block(self, value, name, num_units, subsample, out_channel, learning_mode='normal'):
        for num_unit in xrange(num_units):
            value = self.unit(value, '%s%c' % (name, ord('a') + num_unit), subsample=subsample and num_unit == 0, out_channel=out_channel, learning_mode=learning_mode)
        return value

    def softmax(self, value, dim):
        value = tf.exp(value - tf.reduce_max(value, reduction_indices=dim, keep_dims=True))
        value = value / tf.reduce_sum(value, reduction_indices=dim, keep_dims=True)
        return value

    def segment_mean(self, value):
        shape = image_util.get_shape(value)

        def test_segment_mean(value):
            batch_size = shape[0] / self.num_test_crops
            value = tf.segment_mean(value, np.repeat(np.arange(batch_size), self.num_test_crops))
            return value

        value = self.case([
            (Net.Phase.TRAIN, lambda: value),
            (Net.Phase.TEST, lambda: test_segment_mean(value))])

        value.set_shape((None,) + shape[1:])
        return value


class ResNet50(ResNet):
    def __init__(self,
                 learning_rate=Net.LEARNING_RATE,
                 learning_modes=Net.LEARNING_RATE_MODES,
                 learning_rate_decay_steps=Net.LEARNING_RATE_DECAY_STEPS,
                 learning_rate_decay_rate=Net.LEARNING_RATE_DECAY_RATE,
                 weight_decay=Net.WEIGHT_DECAY,
                 resnet_params_path=ResNet.RESNET_PARAMS_PATH,
                 num_test_crops=ResNet.NUM_TEST_CROPS,
                 is_train=False,
                 is_show=False):

        super(ResNet50, self).__init__(
            learning_rate=learning_rate,
            learning_modes=learning_modes,
            learning_rate_decay_steps=learning_rate_decay_steps,
            learning_rate_decay_rate=learning_rate_decay_rate,
            weight_decay=weight_decay,
            resnet_params_path=resnet_params_path,
            num_test_crops=num_test_crops,
            is_train=is_train,
            is_show=is_show)

    def build(self, blob):
        assert len(blob.as_tuple_list()) == 1, 'Must pass in a single pair of image and label'
        (self.image, self.label) = blob.as_tuple_list()[0]

        with tf.variable_scope('1'):
            self.v0 = self.conv(self.image, 'conv1', size=(7, 7), stride=(2, 2), out_channel=64, biased=True, norm_name='_conv1', activation_fn=tf.nn.relu, learning_mode='slow')
            self.v1 = self.max_pool(self.v0, 'max_pool', size=(3, 3), stride=(2, 2))

        self.v2 = self.block(self.v1, '2', num_units=3, subsample=False, out_channel=64, learning_mode='slow')
        self.v3 = self.block(self.v2, '3', num_units=4, subsample=True, out_channel=128, learning_mode='slow')
        self.v4 = self.block(self.v3, '4', num_units=6, subsample=True, out_channel=256, learning_mode='normal')
        self.v5 = self.block(self.v4, '5', num_units=3, subsample=True, out_channel=512, learning_mode='normal')

        with tf.variable_scope('fc'):
            self.v6 = self.avg_pool(self.v5, 'avg_pool', size=(7, 7))
            self.v7 = self.conv(self.v6, 'fc', out_channel=len(Meta.CLASS_NAMES), biased=True)
            self.v8 = tf.squeeze(self.softmax(self.v7, 3), (1, 2))

        self.feat = self.segment_mean(self.v6)
        self.prob = self.segment_mean(self.v8)

        self.make_stat()

        if self.is_train:
            self.make_train_op()

        if self.is_show:
            self.make_show()

        self.finalize()

    def train(self, iteration, feed_dict=dict()):
        train_dict = dict(train=self.train_op)
        show_dict = self.show_dict[Net.Phase.TRAIN]
        summary_dict = dict(summary=self.summary[Net.Phase.TRAIN])

        self.model.train(
            iteration=iteration,
            feed_dict=util.merge_dicts(feed_dict, {self.phase: Net.Phase.TRAIN.value}),
            callbacks=[
                dict(fetch=util.merge_dicts(train_dict, show_dict, summary_dict)),
                dict(fetch=show_dict,
                     func=lambda **kwargs: self.model.display(begin='Train', end='\n', **kwargs)),
                dict(interval=5,
                     fetch=summary_dict,
                     func=lambda **kwargs: self.model.summary(summary_writer=self.summary_writer, **kwargs)),
                dict(interval=5,
                     func=lambda **kwargs: self.test(feed_dict=feed_dict)),
                dict(interval=1000,
                     func=lambda **kwargs: self.model.save(saver=self.saver, saver_kwargs=dict(save_path=os.path.join(Meta.WORKING_DIR, 'model')), **kwargs))])

    def test(self, iteration=1, feed_dict=dict()):
        show_dict = self.show_dict[Net.Phase.TEST]
        summary_dict = dict(summary=self.summary[Net.Phase.TEST])

        self.model.test(
            iteration=iteration,
            feed_dict=util.merge_dicts(feed_dict, {self.phase: Net.Phase.TEST.value}),
            callbacks=[
                dict(fetch=util.merge_dicts(show_dict, summary_dict)),
                dict(fetch=show_dict,
                     func=lambda **kwargs: self.model.display(begin='\033[2K\rTest', end='\n', **kwargs)),
                dict(fetch=summary_dict,
                     func=lambda **kwargs: self.model.summary(summary_writer=self.summary_writer, **kwargs))])

    def online(self, fetch, feed_dict=dict()):
        feed_dict[self.phase] = Net.Phase.TEST.value

        self.model.test(
            iteration=1,
            feed_dict=feed_dict,
            callbacks=[
                dict(fetch=fetch)])

        return self.model.output_values
