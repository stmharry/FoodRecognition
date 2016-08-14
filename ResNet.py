from __future__ import print_function

import enum
import numpy as np
import os
import scipy
import tensorflow as tf

from deepbox import util, image_util
from deepbox.model import Model


class Meta(object):
    WORKING_DIR = '/tmp/' + str(int(time.time()))
    CLASSNAMES_FILENAME = 'class_names'

    CLASS_NAMES = list()
    NUM_CLASSES = 0


class Input(object):
    @staticmethod
    def get_queue(value_list, dtype):
        queue = tf.FIFOQueue(32, dtypes=[dtype], shapes=[()])
        enqueue = queue.enqueue_many([value_list])
        queue_runner = tf.train.QueueRunner(queue, [enqueue])
        tf.train.add_queue_runner(queue_runner)
        return queue

    def __init__(self, image_dir, is_train=False, subsample_size=1, subsample_inv=False):
        self.filename_list = list()
        self.hash_list = list()
        self.classname_list = list()
        for class_name in os.listdir(image_dir):
            class_dir = os.path.join(image_dir, class_name)
            if class_name.startswith('.') or not os.path.isdir(class_dir):
                continue
            for (file_dir, _, file_names) in os.walk(class_dir):
                for file_name in file_names:
                    if not file_name.endswith('.jpg'):
                        continue
                    hash_ = hash(file_name)
                    if (hash_ % subsample_size == 0) == subsample_inv:
                        continue
                    self.filename_list.append(os.path.join(file_dir, file_name))
                    self.hash_list.append(hash_)
                    self.classname_list.append(class_name)

        self.filename_queue = Input.get_queue(self.filename_list, dtype=tf.string)
        (key, value) = tf.WholeFileReader.read(self.filename_queue)
        self.image = tf.to_float(tf.image.decode_jpeg(value))

        self.classname_queue = Input.get_queue(self.classname_list, dtype=tf.string)
        self.class_name = self.classname_queue.dequeue()

        if is_train:
            class_names = sorted(set(self.classname_list))
            Meta.CLASS_NAMES = class_names
            Meta.NUM_CLASSES = len(class_names)


class ImagePipeline(object):
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
        self.mean = scipy.io.loadmat(mean_path)

    def train(self, image):
        image = tf.to_float(image)
        image = image_util.random_resize(image, size_range=self.train_size_range)
        image = image_util.random_crop(image, size=self.net_size)
        image = image_util.random_flip(image)
        image = image_util.random_adjust_rgb(image)
        image = image - self.mean

        image.set_shape(self.shape)
        return image

    def test_up(self, image):
        image = tf.to_float(image)
        image = tf.tile(tf.expand_dims(image, dim=0), multiples=(self.num_test_crops, 1, 1, 1))

        def test_image_map(image):
            image = image_util.random_resize(image, size_range=self.test_size_range)
            image = image_util.random_crop(image, size=self.net_size)
            image = image_util.random_flip(image)
            image = image - self.mean
            return image

        image = tf.map_fn(test_image_map, image)
        image.set_shape((self.num_test_crops,) + self.shape)
        return image

    def test_down(self, image):
        shape = image_util.get_shape(image)
        image = tf.reshape(image, (shape[0] * shape[1],) + self.shape)
        return image

    def online(self, image):
        return self.test_up(image)


class Batch(object):
    TRAIN_BATCH_SIZE = 64
    TRAIN_CAPACITY = 4096 + 512
    TRAIN_MIN_AFTER_DEQUEUE = 4096
    NUM_TRAIN_THREADS = 8
    TEST_BATCH_SIZE = TRAIN_BATCH_SIZE / ImagePipeline.NUM_TEST_CROPS
    NUM_TEST_THREADS = 1

    def __init__(self,
                 train_batch_size=TRAIN_BATCH_SIZE,
                 train_capacity=TRAIN_CAPACITY,
                 train_min_after_dequeue=TRAIN_MIN_AFTER_DEQUEUE,
                 num_train_threads=NUM_TRAIN_THREADS,
                 test_batch_size=TEST_BATCH_SIZE,
                 num_test_threads=NUM_TEST_THREADS):

        self.train_batch_size = train_batch_size
        self.train_capacity = train_capacity
        self.train_min_after_dequeue = train_min_after_dequeue
        self.num_train_threads = num_train_threads
        self.test_batch_size = test_batch_size
        self.num_test_threads = num_test_threads

    def train(self, args_list):
        return tf.train.shuffle_batch_join(
            args_list,
            batch_size=self.train_batch_size,
            num_threads=self.num_train_threads,
            capacity=self.train_capacity,
            min_after_dequeue=self.train_min_after_dequeue)

    def test(self, args_list):
        return tf.train.batch_join(
            args_list,
            batch_size=self.test_batch_size,
            num_threads=self.num_test_threads,
            capacity=self.test_batch_size)


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
                 image,
                 class_name=str(),
                 learning_rate=LEARNING_RATE,
                 learning_modes=LEARNING_MODES,
                 weight_decay=WEIGHT_DECAY,
                 working_dir=WORKING_DIR,
                 classnames_filename=CLASSNAMES_FILENAME,
                 is_train=False,
                 is_show=False):

        self.image = image
        self.class_name = class_name
        self.learning_rate = Net.get_const_variable(learning_rate, 'learning_rate')
        self.learning_modes = learning_modes
        self.weight_decay = weight_decay

        self.working_dir = working_dir
        self.checkpoint = tf.train.get_checkpoint_state(working_dir)
        classnames_path = os.path.join(working_dir, classnames_filename)
        self.is_train = is_train
        self.is_show = is_show

        self.phase = Net.placeholder('phase', shape=())
        if self.checkpoint:
            self.class_names = np.loadtxt(classnames_path, dtype=np.str, delimiter=',')
        else:
            np.savetxt(classnames_path, Net.CLASS_NAMES, delimiter=',', fmt='%s')
        self.class_names = Net.get_const_variable(Input.CLASS_NAMES, 'class_names', shape=(Input.NUM_CLASSES,), dtype=tf.string)
        self.global_step = Net.get_const_variable(0, 'global_step')

    def make_stat(self):
        assert hasattr(self, 'prob')

        self.target = tf.equal(self.class_names, self.class_name)
        self.target_frac = tf.reduce_mean(self.target, 0)
        self.loss = - tf.reduce_mean(self.target * tf.log(self.prob + util.EPSILON)) * Input.NUM_CLASSES
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        if regularization_losses:
            self.loss += tf.add_n(regularization_losses)

        self.label = tf.argmax(self.target, 1)
        self.pred_label = tf.argmax(self.prob, 1)
        self.pred_classname = tf.gather(self.class_names, self.pred_label)
        self.correct = tf.to_float(tf.equal(self.label, self.pred_label))
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
        def identity(value):
            return value

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
        if Net.CHECKPOINT:
            print('Model restored from %s' % Net.CHECKPOINT.model_checkpoint_path)
            self.saver.restore(tf.get_default_session(), Net.CHECKPOINT.model_checkpoint_path)

        tf.train.start_queue_runners()
        print('Filling queues with images...')
        self.model = Model(self.global_step)


class ResNet(Net):
    RESNET_PARAMS_PATH = 'archive/ResNet-50-params.mat'

    def __init__(self,
                 image,
                 class_name=None,
                 resnet_params_path=RESNET_PARAMS_PATH,
                 working_dir=None,
                 is_train=False,
                 is_show=False):

        super(ResNet, self).__init__(
            image,
            class_name,
            working_dir=working_dir,
            is_train=is_train,
            is_show=is_show)
        self.resnet_params_path = resnet_params_path
        self.mean_path = self.mean_path

        if not self.checkpoint:
            self.resnet_params = scipy.io.loadmat(self.resnet_params_path)

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

        if Net.LEARNING_MODES[learning_mode] > 0:
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

    def segment_mean(self, value, phase):
        # TODO: move to ImagePipeline
        def test_segment_mean(value):
            shape = image_util.get_shape(value)
            reduced_batch_size = shape[0] / ImagePipeline.NUM_TEST_CROPS
            value = tf.segment_mean(value, np.repeat(np.arange(reduced_batch_size), ImagePipeline.NUM_TEST_CROPS))
            value.set_shape((reduced_batch_size,) + shape[1:])
            return value

        value = tf.case([
            (tf.equal(phase, Net.Phase.TRAIN), lambda: value),
            (tf.equal(phase, Net.Phase.TEST), lambda: test_segment_mean(value))], default=lambda: value)


class ResNet50(ResNet):
    def __init__(self,
                 image,
                 class_name=None,
                 resnet_params_path=None,
                 working_dir=None,
                 is_train=False,
                 is_show=False):

        super(ResNet50, self).__init__(
            image,
            class_name,
            resnet_params_path=resnet_params_path,
            working_dir=working_dir,
            is_train=is_train,
            is_show=is_show)

        with tf.variable_scope('1'):
            self.v0 = sele.conv(self.image, 'conv1', size=(7, 7), stride=(2, 2), out_channel=64, biased=True, norm_name='_conv1', activation_fn=tf.nn.relu, learning_mode='slow')
            self.v1 = self.max_pool(self.v0, 'max_pool', size=(3, 3), stride=(2, 2))

        self.v2 = self.block(self.v1, '2', num_units=3, subsample=False, out_channel=64, learning_mode='slow')
        self.v3 = self.block(self.v2, '3', num_units=4, subsample=True, out_channel=128, learning_mode='slow')
        self.v4 = self.block(self.v3, '4', num_units=6, subsample=True, out_channel=256, learning_mode='normal')
        self.v5 = self.block(self.v4, '5', num_units=3, subsample=True, out_channel=512, learning_mode='normal')

        with tf.variable_scope('fc'):
            self.v6 = self.avg_pool(self.v5, 'avg_pool', size=(7, 7))
            self.v7 = self.conv(self.v6, 'fc', out_channel=Input.NUM_CLASSES, biased=True)
            self.v8 = tf.squeeze(self.softmax(self.v7, 3), (1, 2))

        self.feat = self.segment_mean(self.v6, self.phase)
        self.prob = self.segment_mean(self.v8, self.phase)

        self.make_stat()

        if self.is_train:
            self.make_train_op()

        if self.is_show:
            self.make_show()

        self.finalize()

    def train(self, iteration, feed_dict=dict()):
        updates_per_iteration = util.merge_dicts(
            dict(
                train=self.train_op,
                summary=self.summary),
            self.show_dict[Net.Phase.TRAIN])

        feed_dict[self.phase] = Net.Phase.TRAIN

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

    def test(self):
        pass  # TODO

    def online(self, fetch, feed_dict=dict()):
        feed_dict[self.phase] = Net.Phase.TEST

        self.model.test(
            iteration=1,
            feed_dict=feed_dict,
            callbacks=[
                dict(fetch=fetch)])

        return self.model.output_values
