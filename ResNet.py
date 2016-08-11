from __future__ import print_function

import enum
import numpy as np
import os
import scipy
import tensorflow as tf

import data

from deepbox import util, image_util
from deepbox.model import Model

RESNET_MAT_PATH = '/mnt/data/ResNet-50-params.mat'  # TODO
RESNET_MAT = scipy.io.loadmat(RESNET_MAT_PATH)
WORKING_DIR = ''  # TODO


def identity(value):
    return value


class Image(object):
    NUM_TEST_CROPS = 4
    TRAIN_SIZE_RANGE = (256, 512)
    TEST_SIZE_RANGE = (384, 384)
    NET_SIZE = 224
    NET_CHANNEL = 3

    @staticmethod
    def train(image):
        image = tf.to_float(image)
        image = image_util.random_resize(image, size_range=Image.TRAIN_SIZE_RANGE)
        image = image_util.random_crop(image, size=Image.NET_SIZE)
        image = image_util.random_flip(image)
        image = image_util.random_adjust_rgb(image)
        image = image - RESNET_MAT['mean']

        image.set_shape((Image.NET_SIZE, Image.NET_SIZE, Image.NET_CHANNEL))
        return image

    @staticmethod
    def test_up(image):
        image = tf.to_float(image)
        image = tf.tile(tf.expand_dims(image, dim=0), multiples=(Image.NUM_TEST_CROPS, 1, 1, 1))

        def test_image_map(image):
            image = image_util.random_resize(image, size_range=Image.TEST_SIZE_RANGE)
            image = image_util.random_crop(image, size=Image.NET_SIZE)
            image = image_util.random_flip(image)
            image = image - RESNET_MAT['mean']
            return image

        image = tf.map_fn(test_image_map, image)
        image.set_shape((Image.NUM_TEST_CROPS, Image.NET_SIZE, Image.NET_SIZE, Image.NET_CHANNEL))
        return image

    @staticmethod
    def test_down(image):
        shape = image_util.get_shape(image)
        image = tf.reshape(image, (shape[0] * shape[1], Image.NET_SIZE, Image.NET_SIZE, Image.NET_CHANNEL))
        return image

    @staticmethod
    def online(image):
        return Image.test_up(image)


class Batch(object):
    TRAIN_BATCH_SIZE = 64
    TRAIN_CAPACITY = 4096 + 512
    TRAIN_MIN_AFTER_DEQUEUE = 4096
    TEST_BATCH_SIZE = TRAIN_BATCH_SIZE / Image.NUM_TEST_CROPS
    ONLINE_BATCH_SIZE = 1

    @staticmethod
    def train(args_list):
        args = tf.train.shuffle_batch_join(
            args_list,
            batch_size=Batch.TRAIN_BATCH_SIZE,
            capacity=Batch.TRAIN_CAPACITY,
            min_after_dequeue=Batch.TRAIN_MIN_AFTER_DEQUEUE)
        return args

    @staticmethod
    def test(args_list):
        args = tf.train.batch_join(
            args_list,
            batch_size=Batch.TEST_BATCH_SIZE,
            capacity=Batch.TEST_BATCH_SIZE)
        return args


class Net(object):
    class Phase(enum.Enum):
        TRAIN = 0
        TEST = 1

    NET_VARIABLES = 'net_variables'
    NET_COLLECTIONS = [tf.GraphKeys.VARIABLES, NET_VARIABLES]

    LEARNING_RATE = 1e-1
    LEARNING_MODES = dict(normal=1.0, slow=0.0)
    WEIGHT_DECAY = 0.0

    @staticmethod
    def placeholder(name, shape=(), dtype=tf.float32):
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

    def __init__(self, phase, image, label, is_train=False, is_show=False):
        self.phase = phase
        self.image = image
        self.label = tf.to_int64(label)
        self.is_train = is_train
        self.is_show = is_show

    def make_stat(self):
        assert hasattr(self, 'logit')

        self.target = tf.one_hot(self.label, len(data.CLASS_NAMES))
        self.target_frac = tf.reduce_mean(self.target, 0)
        self.loss = - tf.reduce_mean(self.target * tf.log(self.logit + util.EPSILON)) * len(data.CLASS_NAMES)
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        if regularization_losses:
            self.loss += tf.add_n(regularization_losses)

        self.pred = tf.argmax(self.logit, 1)
        self.pred_names = tf.gather(self.class_names, self.pred)
        self.correct = tf.to_float(tf.equal(self.pred, self.label))
        self.correct_frac = tf.reduce_mean(tf.expand_dims(self.correct, 1) * self.target, 0)

    def make_train_op(self):
        train_ops = []
        for (learning_mode, learning_rate_relative) in Net.LEARNING_MODES.iteritems():
            variables = tf.get_collection(learning_mode)
            if variables:
                train_ops.append(tf.train.AdamOptimizer(
                    learning_rate=self.learning_rate * learning_rate_relative,
                    epsilon=1.0).minimize(self.loss, global_step=self.global_step))
        self.train_op = tf.group(*train_ops)

    def make_show(self):
        postfix_funcs = {
            Net.Phase.TRAIN: {
                'raw': identity,
                'avg': lambda value: util.exponential_moving_average(value, num_updates=self.global_step)},
            Net.Phase.TEST: {
                'raw': identity}}

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
        self.summary_writer = tf.train.SummaryWriter(WORKING_DIR)

        self.sess.run(tf.initialize_all_variables())
        checkpoint = tf.train.get_checkpoint_state(WORKING_DIR)
        if checkpoint:
            print('[ Model restored from %s ]' % checkpoint.model_checkpoint_path)
            self.saver.restore(tf.get_default_session(), checkpoint.model_checkpoint_path)

        tf.train.start_queue_runners()
        print('[ Filling queues with images ]')
        self.model = Model(self.global_step)


class ResNet(Net):
    @staticmethod
    def conv(value, conv_name, out_channel, size=(1, 1), stride=(1, 1), padding='SAME', biased=False, norm_name=None, activation_fn=None, learning_mode='normal'):
        in_channel = image_util.get_channel(value)

        if Net.LEARNING_MODES[learning_mode] > 0:
            collections = Net.NET_COLLECTIONS + [learning_mode]
            trainable = True
        else:
            collections = Net.NET_COLLECTIONS
            trainable = False

        if conv_name in RESNET_MAT:
            print('%s initialized from %s' % (conv_name, RESNET_MAT_PATH))
            weights_initializer = tf.constant_initializer(RESNET_MAT[conv_name][0][0])
        else:
            print('%s initialized randomly' % conv_name)
            weights_initializer = tf.truncated_normal_initializer(stddev=(2. / (in_channel * stride[0] * stride[1])) ** 0.5)

        if Net.WEIGHT_DECAY > 0:
            weight_regularizer = tf.contrib.layers.l2_regularizer(Net.WEIGHT_DECAY)
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
            if conv_name in RESNET_MAT:
                bias_initializer = tf.constant_initializer(RESNET_MAT[conv_name][1][0][:, 0])
            else:
                bias_initializer = tf.constant_initializer(0.1)

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

            if bn_name in RESNET_MAT:
                print('%s initialized from %s' % (bn_name, RESNET_MAT_PATH))
                mean_initializer = tf.constant_initializer(RESNET_MAT[bn_name][0][0][:, 0])
                variance_initializer = tf.constant_initializer(RESNET_MAT[bn_name][1][0][:, 0])
            else:
                print('%s initialized with constant' % bn_name)
                mean_initializer = tf.constant_initializer(0.0)
                variance_initializer = tf.constant_initializer(1.0)

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

            if scale_name in RESNET_MAT:
                print('%s initialized from %s' % (scale_name, RESNET_MAT_PATH))
                scale_initializer = tf.constant_initializer(RESNET_MAT[scale_name][0][0][:, 0])
                offset_initializer = tf.constant_initializer(RESNET_MAT[scale_name][1][0][:, 0])
            else:
                print('%s initialized with constant' % scale_name)
                scale_initializer = tf.constant_initializer(1.0)
                offset_initializer = tf.constant_initializer(0.0)

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

    @staticmethod
    def unit(value, name, subsample, out_channel, learning_mode='normal'):
        in_channel = image_util.get_channel(value)

        if subsample:
            stride = (2, 2)
        else:
            stride = (1, 1)

        out_channel_inner = out_channel
        out_channel_outer = 4 * out_channel

        with tf.variable_scope(name):
            if subsample or in_channel != out_channel_outer:
                value1 = ResNet.conv(value, 'res%s_branch1' % name, out_channel=out_channel_outer, stride=stride, norm_name='%s_branch1' % name, learning_mode=learning_mode)
            else:
                value1 = value

            value2 = ResNet.conv(value, 'res%s_branch2a' % name, out_channel=out_channel_inner, stride=stride, norm_name='%s_branch2a' % name, activation_fn=tf.nn.relu, learning_mode=learning_mode)
            value2 = ResNet.conv(value2, 'res%s_branch2b' % name, out_channel=out_channel_inner, size=(3, 3), norm_name='%s_branch2b' % name, activation_fn=tf.nn.relu, learning_mode=learning_mode)
            value2 = ResNet.conv(value2, 'res%s_branch2c' % name, out_channel=out_channel_outer, norm_name='%s_branch2c' % name, learning_mode=learning_mode)

            value = tf.nn.relu(value1 + value2)
        return value

    @staticmethod
    def block(value, name, num_units, subsample, out_channel, learning_mode='normal'):
        for num_unit in xrange(num_units):
            value = ResNet.unit(value, '%s%c' % (name, ord('a') + num_unit), subsample=subsample and num_unit == 0, out_channel=out_channel, learning_mode=learning_mode)
        return value

    @staticmethod
    def softmax(value, dim):
        value = tf.exp(value - tf.reduce_max(value, reduction_indices=dim, keep_dims=True))
        value = value / tf.reduce_sum(value, reduction_indices=dim, keep_dims=True)
        return value

    @staticmethod
    def segment_mean(self, value, phase):
        def test_segment_mean(value):
            shape = image_util.get_shape(value)
            reduced_batch_size = shape[0] / Image.NUM_TEST_CROPS
            value = tf.segment_mean(value, np.repeat(np.arange(reduced_batch_size), Image.NUM_TEST_CROPS))
            value.set_shape((reduced_batch_size,) + shape[1:])
            return value

        value = tf.case([
            (tf.equal(phase, Net.Phase.TRAIN), lambda: value),
            (tf.equal(phase, Net.Phase.TEST), lambda: test_segment_mean(value))], default=lambda: value)

    def __init__(self, phase, image, label, is_train=False, is_show=False):
        super(ResNet, self).__init__(phase, image, label, is_train=is_train, is_show=is_show)


class ResNet50(ResNet):
    def __init__(self, phase, image, label, is_train=False, is_show=False):
        super(ResNet50, self).__init__(phase, image, label, is_train=is_train, is_show=is_show)

        self.class_names = ResNet.get_const_variable(data.CLASS_NAMES, 'class_names', shape=(len(data.CLASS_NAMES),), dtype=tf.string)
        self.global_step = ResNet.get_const_variable(0, 'global_step')
        self.learning_rate = ResNet.get_const_variable(ResNet.LEARNING_RATE, 'learning_rate')

        with tf.variable_scope('1'):
            self.v0 = ResNet.conv(self.image, 'conv1', size=(7, 7), stride=(2, 2), out_channel=64, biased=True, norm_name='_conv1', activation_fn=tf.nn.relu, learning_mode='slow')
            self.v1 = ResNet.max_pool(self.v0, 'max_pool', size=(3, 3), stride=(2, 2))

        self.v2 = ResNet.block(self.v1, '2', num_units=3, subsample=False, out_channel=64, learning_mode='slow')
        self.v3 = ResNet.block(self.v2, '3', num_units=4, subsample=True, out_channel=128, learning_mode='slow')
        self.v4 = ResNet.block(self.v3, '4', num_units=6, subsample=True, out_channel=256, learning_mode='normal')
        self.v5 = ResNet.block(self.v4, '5', num_units=3, subsample=True, out_channel=512, learning_mode='normal')

        with tf.variable_scope('fc'):
            self.v6 = ResNet.avg_pool(self.v5, 'avg_pool', size=(7, 7))
            self.v7 = ResNet.conv(self.v6, 'fc', out_channel=len(data.CLASS_NAMES), biased=True)
            self.v8 = tf.squeeze(ResNet.softmax(self.v7, 3), (1, 2))

        self.feat = ResNet.segment_mean(self.v6, self.phase)
        self.logit = ResNet.segment_mean(self.v8, self.phase)

        self.make_stat()

        if self.is_train:
            self.make_train_op()

        if self.is_show:
            self.make_show()

        self.finalize()

    def train(self, iteration, feed_dict):
        updates_per_iteration = util.merge_dicts(
            dict(
                train=self.train_op,
                summary=self.summary),
            self.show_dict[Net.Phase.TRAIN])

        self.model.train(
            iteration=iteration,
            feed_dict=feed_dict,
            callbacks=[
                dict(fetch=updates_per_iteration),
                dict(fetch=self.show_dict[Net.Phase.TRAIN],
                     func=lambda **kwargs: self.model.display(begin='Train', end='\n', **kwargs)),
                dict(interval=5,
                     fetch=dict(summary=self.summary),
                     func=lambda **kwargs: self.model.summary(summary_writer=self.summary_writer, **kwargs)),
                dict(interval=1000,
                     func=lambda **kwargs: self.model.save(saver=self.saver, saver_kwargs=dict(save_path=os.path.join(WORKING_DIR, 'model')), **kwargs))])

    def test(self, feed_dict, fetch):
        self.model.test(
            iteration=1,
            feed_dict=feed_dict,
            callbacks=[
                dict(fetch=fetch)])
