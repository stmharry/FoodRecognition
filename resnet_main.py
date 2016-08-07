#!/usr/bin/env python
'''Trains Residual Network-50 Layers for Food Classification.

Enter "./resnet_main.py -h" for a list of usage
'''

from __future__ import print_function

import numpy as np
import os
import tensorflow as tf

import data

from deepbox import util, image_util
from deepbox.model import Model

__author__ = 'Tzu-Ming Harry Hsu'
__email__ = 'harry19930924@gmail.com'

tf.app.flags.DEFINE_integer('batch_size', 64, 'mini-batch size (over all GPUs). [default: 64]')
tf.app.flags.DEFINE_integer('num_train_pipelines', 8, 'number of parallel training pipelines. [default: 8]')
tf.app.flags.DEFINE_integer('num_test_pipelines', 1, 'number of parallel testing pipelines. [default: 1]')
tf.app.flags.DEFINE_integer('num_test_crops', 1, 'random cropping at testing time. [default: 1]')
tf.app.flags.DEFINE_integer('num_gpus', 1, 'number of GPUs used. [default: 1]')
tf.app.flags.DEFINE_integer('train_iteration', 10000, 'number of iterations if used with --command=train. [default: 10000]')
tf.app.flags.DEFINE_integer('lr_half_per', 1500, 'learning rate half life. [default: 1500]')
tf.app.flags.DEFINE_float('subsample_ratio', 1.0, 'subsample ratio for training images. [default: 1.0')
tf.app.flags.DEFINE_float('lr', 1e-1, 'initial learning rate. [default: 1e-1]')
tf.app.flags.DEFINE_float('lr_slow', 0.0, 'relative learning rate for shallow layers. (so its learning rate would be (lr * lr_slow). [default: 0.0]')
tf.app.flags.DEFINE_float('weight_decay', 0.0, 'weight decay for convolutional kernels. [default: 0.0]')
tf.app.flags.DEFINE_string('command', 'none', '"train" to run the pre-schuduled training scheme, "test" to run tests and generate log file, "online" to run as a server, and "none" to do nothing. [default: none]')
tf.app.flags.DEFINE_string('log_file', '/tmp/test_log.csv', 'log file name for test, used with --command=test. [default: /tmp/test_log.csv]')
tf.app.flags.DEFINE_string('working_dir', '/tmp', 'working directory for saving logs and models. [default: /tmp]')
tf.app.flags.DEFINE_string('train_dir', None, 'directory for files to be trained.')
tf.app.flags.DEFINE_string('test_dir', None, 'directory for files to be tested, leave blank for validation split. [default: none]')
tf.app.flags.DEFINE_string('test_attrs', 'key,pred_name', 'names for test logging.')
FLAGS = tf.app.flags.FLAGS
FLAGS._parse_flags()

if FLAGS.batch_size == -1:
    FLAGS.batch_size = FLAGS.num_test_crops

LEARNING_RATE_RELATIVE_DICT = dict(normal=1.0, slow=FLAGS.lr_slow)
NET_VARIABLES = 'net_variables'

TRAIN = 0
TEST = 1
TEST_ALT = 2
PHASES = [TRAIN, TEST]
NAMES = {TRAIN: 'train', TEST: 'test'}


def avg_pool(value, name, size, stride=None, padding='SAME'):
    with tf.variable_scope(name):
        if stride is None:
            stride = size
        value = tf.nn.avg_pool(value, ksize=(1,) + size + (1,), strides=(1,) + stride + (1,), padding=padding, name='pool')
    return value


def max_pool(value, name, size, stride=None, padding='SAME'):
    with tf.variable_scope(name):
        if stride is None:
            stride = size
        value = tf.nn.max_pool(value, ksize=(1,) + size + (1,), strides=(1,) + stride + (1,), padding=padding, name='pool')
    return value


def conv(value, conv_name, out_channel, size=(1, 1), stride=(1, 1), padding='SAME', biased=False, norm_name=None, activation_fn=None, learning_mode='normal'):
    in_channel = image_util.get_channel(value)

    if LEARNING_RATE_RELATIVE_DICT[learning_mode] > 0:
        collections = [tf.GraphKeys.VARIABLES, NET_VARIABLES, learning_mode]
        trainable = True
    else:
        collections = [tf.GraphKeys.VARIABLES, NET_VARIABLES]
        trainable = False

    if conv_name in data.RESNET_MAT:
        weights_initializer = tf.constant_initializer(data.RESNET_MAT[conv_name][0][0])
    else:
        print('%s not found' % conv_name)
        weights_initializer = tf.truncated_normal_initializer(stddev=(2. / (in_channel * stride[0] * stride[1])) ** 0.5)

    if FLAGS.weight_decay > 0:
        weight_regularizer = tf.contrib.layers.l2_regularizer(FLAGS.weight_decay)
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

    value = tf.nn.conv2d(value, weight, strides=(1,) + stride + (1,), padding=padding)

    if biased:
        if conv_name in data.RESNET_MAT:
            bias_initializer = tf.constant_initializer(data.RESNET_MAT[conv_name][1][0][:, 0])
        else:
            bias_initializer = tf.constant_initializer(0.1)

        with tf.variable_scope(conv_name):
            bias = tf.get_variable(
                'bias',
                shape=(out_channel),
                initializer=bias_initializer,
                trainable=trainable,
                collections=collections)

        value = tf.nn.bias_add(value, bias)

    if norm_name is not None:
        bn_name = 'bn%s' % norm_name
        scale_name = 'scale%s' % norm_name

        if bn_name in data.RESNET_MAT:
            mean_initializer = tf.constant_initializer(data.RESNET_MAT[bn_name][0][0][:, 0])
            variance_initializer = tf.constant_initializer(data.RESNET_MAT[bn_name][1][0][:, 0])
        else:
            print('%s not found' % bn_name)
            mean_initializer = tf.constant_initializer(0.0)
            variance_initializer = tf.constant_initializer(1.0)

        with tf.variable_scope(bn_name):
            mean = tf.get_variable(
                'mean',
                shape=(out_channel,),
                classeinitializer=mean_initializer,
                trainable=False,
                collections=[tf.GraphKeys.VARIABLES, NET_VARIABLES])
            variance = tf.get_variable(
                'variance',
                shape=(out_channel,),
                initializer=variance_initializer,
                trainable=False,
                collections=[tf.GraphKeys.VARIABLES, NET_VARIABLES])

        if scale_name in data.RESNET_MAT:
            scale_initializer = tf.constant_initializer(data.RESNET_MAT[scale_name][0][0][:, 0])
            offset_initializer = tf.constant_initializer(data.RESNET_MAT[scale_name][1][0][:, 0])
        else:
            print('%s not found' % scale_name)
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
            value1 = conv(value, 'res%s_branch1' % name, out_channel=out_channel_outer, stride=stride, norm_name='%s_branch1' % name, learning_mode=learning_mode)
        else:
            value1 = value

        value2 = conv(value, 'res%s_branch2a' % name, out_channel=out_channel_inner, stride=stride, norm_name='%s_branch2a' % name, activation_fn=tf.nn.relu, learning_mode=learning_mode)
        value2 = conv(value2, 'res%s_branch2b' % name, out_channel=out_channel_inner, size=(3, 3), norm_name='%s_branch2b' % name, activation_fn=tf.nn.relu, learning_mode=learning_mode)
        value2 = conv(value2, 'res%s_branch2c' % name, out_channel=out_channel_outer, norm_name='%s_branch2c' % name, learning_mode=learning_mode)

        value = tf.nn.relu(value1 + value2)
    return value


def block(value, name, num_units, subsample, out_channel, learning_mode='normal'):
    for n in xrange(num_units):
        value = unit(value, '%s%c' % (name, ord('a') + n), subsample=subsample and n == 0, out_channel=out_channel, learning_mode=learning_mode)
    return value


def softmax(value, dim):
    value = tf.exp(value - tf.reduce_max(value, reduction_indices=dim, keep_dims=True))
    value = value / tf.reduce_sum(value, reduction_indices=dim, keep_dims=True)
    return value


def segment_mean(value):
    shape = image_util.get_shape(value)
    test_batch_size_per_gpu = (FLAGS.batch_size / FLAGS.num_test_crops) / FLAGS.num_gpus
    value = tf.segment_mean(value, np.repeat(np.arange(test_batch_size_per_gpu), FLAGS.num_test_crops))
    value.set_shape((None,) + shape[1:])
    return value


''' Parameters
'''
phase = tf.placeholder(
    name='phase',
    shape=(),
    dtype=tf.int32)

global_step = tf.get_variable(
    'global_step',
    shape=(),
    initializer=tf.constant_initializer(),
    trainable=False,
    collections=[tf.GraphKeys.VARIABLES, NET_VARIABLES])

learning_rate = tf.get_variable(
    'learning_rate',
    shape=(),
    initializer=tf.constant_initializer(FLAGS.lr),
    trainable=False,
    collections=[tf.GraphKeys.VARIABLES, NET_VARIABLES])

if FLAGS.command == 'train':
    learning_rate = tf.train.exponential_decay(
        learning_rate=learning_rate,
        global_step=global_step,
        decay_steps=FLAGS.lr_half_per,
        decay_rate=0.5,
        staircase=True)

''' Pipeline
'''
data.cache_train_files(directory=FLAGS.train_dir)

if FLAGS.command == 'train' or FLAGS.command == 'none':
    train_files = data.get_files(data.DEV, num_pipelines=FLAGS.num_train_pipelines, subsample_ratio=FLAGS.subsample_ratio, directory=FLAGS.train_dir)
    train_values = data.get_train_values(
        train_files,
        batch_size=FLAGS.batch_size)

if FLAGS.command == 'train':
    test_files = data.get_files(data.VAL, num_pipelines=FLAGS.num_test_pipelines, directory=FLAGS.train_dir)
elif FLAGS.command == 'test':
    test_files = data.get_files(data.TEST, num_pipelines=FLAGS.num_test_pipelines, directory=FLAGS.test_dir)
elif
test_values = data.get_test_values(
    test_files,
    batch_size=FLAGS.batch_size,
    num_test_crops=FLAGS.num_test_crops)
test_batch_size = FLAGS.batch_size / FLAGS.num_test_crops
test_iteration = sum(map(len, test_files)) / test_batch_size

class_names = tf.get_variable(
    'class_names',
    shape=(len(data.CLASS_NAMES),),
    dtype=tf.string,
    initializer=tf.constant_initializer(data.CLASS_NAMES),
    trainable=False,
    collections=[tf.GraphKeys.VARIABLES, NET_VARIABLES])

if FLAGS.command == 'test':
    (key, image, label) = test_values
else:
    (key, image, label) = tf.case([
        (tf.equal(phase, TRAIN), lambda: train_values),
        (tf.equal(phase, TEST), lambda: test_values)], default=lambda: train_values)

key.set_shape((None,))
image.set_shape((None, data.SIZE, data.SIZE, 3))
label.set_shape((None,))


''' Main Network
'''
image_splits = tf.split(0, FLAGS.num_gpus, image)
label_splits = tf.split(0, FLAGS.num_gpus, label)
logit_splits = []
loss_splits = []
grads_splits = []

for num_gpu in xrange(FLAGS.num_gpus):
    with tf.device('/gpu:%d' % num_gpu):
        with tf.name_scope('gpu-%d' % num_gpu):
            value = image_splits[num_gpu]

            with tf.variable_scope('1'):
                value = conv(value, 'conv1', size=(7, 7), stride=(2, 2), out_channel=64, biased=True, norm_name='_conv1', activation_fn=tf.nn.relu, learning_mode='slow')
                value = max_pool(value, 'max-pool', size=(3, 3), stride=(2, 2))
            value1 = value
            value = block(value, '2', num_units=3, subsample=False, out_channel=64, learning_mode='slow')
            value2 = value
            value = block(value, '3', num_units=4, subsample=True, out_channel=128, learning_mode='slow')
            value3 = value
            value = block(value, '4', num_units=6, subsample=True, out_channel=256)
            value4 = value
            value = block(value, '5', num_units=3, subsample=True, out_channel=512)
            value5 = value

            with tf.variable_scope('fc'):
                value = avg_pool(value, 'avg-pool', size=(7, 7))
                size = image_util.get_size(value)
                feat = tf.case([
                    (tf.equal(phase, TRAIN), lambda: value),
                    (tf.equal(phase, TEST), lambda: segment_mean(value))], default=lambda: value)
                feat.set_shape((None,) + size)
                value = conv(value, 'fc', out_channel=len(data.CLASS_NAMES), biased=True)

                value = tf.squeeze(softmax(value, dim=3), squeeze_dims=(1, 2))
                size = image_util.get_size(value)
                logit = tf.case([
                    (tf.equal(phase, TRAIN), lambda: value),
                    (tf.equal(phase, TEST), lambda: segment_mean(value))], default=lambda: value)
                logit.set_shape((None,) + size)

            target = tf.one_hot(label_splits[num_gpu], len(data.CLASS_NAMES))
            loss = - tf.reduce_mean(target * tf.log(logit + util.EPSILON)) * len(data.CLASS_NAMES)
            regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            if regularization_losses:
                loss += tf.add_n(regularization_losses)
            grads = tf.gradients(loss, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

            logit_splits.append(logit)
            loss_splits.append(loss)
            grads_splits.append(grads)

            tf.get_variable_scope().reuse_variables()

logit = tf.concat(0, logit_splits)
prediction = tf.argmax(logit, 1)
pred_name = tf.gather(class_names, prediction)
correct = tf.to_float(tf.equal(prediction, label))

acc = tf.reduce_mean(correct)
loss = tf.reduce_mean(tf.pack(loss_splits))

target = tf.one_hot(label, len(data.CLASS_NAMES))
target_frac = tf.reduce_mean(target, 0)
correct_frac = tf.reduce_mean(tf.expand_dims(correct, 1) * target, 0)

grads = [tf.reduce_mean(tf.pack(grads_for_var), reduction_indices=0) for grads_for_var in zip(*grads_splits)]
var_grads = dict(zip(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), grads))

train_ops = [global_step.assign_add(1)]
for (learning_mode, learning_rate_relative) in LEARNING_RATE_RELATIVE_DICT.iteritems():
    variables = tf.get_collection(learning_mode)
    if variables:
        grad_vars = [(var_grads[var], var) for var in variables]
        train_ops.append(tf.train.AdamOptimizer(
            learning_rate=learning_rate * learning_rate_relative,
            epsilon=1.0).apply_gradients(grad_vars))
train_op = tf.group(*train_ops)

postfix_funcs = {
    TRAIN: {
        'raw': lambda x: x,
        'avg': lambda x: util.exponential_moving_average(x, num_updates=global_step)},
    TEST: {
        'avg': lambda x: util.moving_average(x, window=test_iteration)}}

display_dict = {
    TRAIN: {attr: globals()[attr] for attr in ['learning_rate']},
    TEST: {}}

for p in PHASES:
    for (postfix, func) in postfix_funcs[p].iteritems():
        for attr in ['loss', 'acc']:
            display_dict[p].update({
                '%s_%s_%s' % (NAMES[p], attr, postfix): func(globals()[attr])})

summary_dict = {k: v.copy() for (k, v) in display_dict.iteritems()}

for p in PHASES:
    (postfix, func) = ('avg', postfix_funcs[p]['avg'])
    accs = func(correct_frac) / (func(target_frac) + util.EPSILON)
    for (i, class_name) in enumerate(data.CLASS_NAMES):
        summary_dict[p].update({
            '%s_acc(%s)_%s' % (NAMES[p], class_name, postfix): accs[i]})

summary = {
    p: tf.merge_summary([tf.scalar_summary(k, v) for (k, v) in summary_dict[p].iteritems()])
    for p in PHASES}

sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
saver = tf.train.Saver(tf.get_collection(NET_VARIABLES), keep_checkpoint_every_n_hours=0.5)
summary_writer = tf.train.SummaryWriter(FLAGS.working_dir)

sess.run(tf.initialize_all_variables())
checkpoint = tf.train.get_checkpoint_state(FLAGS.working_dir)
if checkpoint:
    print('[ Model restored from %s ]' % checkpoint.model_checkpoint_path)
    saver.restore(tf.get_default_session(), checkpoint.model_checkpoint_path)

tf.train.start_queue_runners()
print('[ Filling queues with images ]')
model = Model(global_step)


def lr(new_learning_rate):
    sess.run(learning_rate.assign(new_learning_rate))


def train(iteration=FLAGS.train_iteration):
    model.train(
        iteration=iteration,
        feed_dict={phase: TRAIN},
        callbacks=[
            dict(fetch=dict(train=train_op)),
            dict(fetch=display_dict[TRAIN],
                 func=lambda **kwargs: model.display(begin='Train', end='\n', **kwargs)),
            dict(fetch=summary_dict[TRAIN]),
            dict(interval=5,
                 fetch=dict(summary=summary[TRAIN]),
                 func=lambda **kwargs: model.summary(summary_writer=summary_writer, **kwargs)),
            dict(interval=1000,
                 func=lambda **kwargs: model.save(saver=saver, saver_kwargs=dict(save_path=os.path.join(FLAGS.working_dir, 'model')), **kwargs)),
            dict(interval=500,
                 func=lambda **kwargs: val())])


def val():
    model.test(
        iteration=test_iteration,
        feed_dict={phase: TEST},
        callbacks=[
            dict(fetch=display_dict[TEST],
                 func=lambda **kwargs: model.display(begin='\033[2K\rVal', end='', **kwargs)),
            dict(fetch=summary_dict[TEST]),
            dict(interval=-1,
                 func=lambda **kwargs: print('')),
            dict(interval=-1,
                 fetch=dict(summary=summary[TEST]),
                 func=lambda **kwargs: model.summary(summary_writer=summary_writer, **kwargs))])


def test(log_file=FLAGS.log_file, attrs=FLAGS.test_attrs.split(',')):
    import csv
    np.set_printoptions(threshold=np.nan, linewidth=np.inf)

    log_handle = open(log_file, 'w')
    writer = csv.writer(log_handle, lineterminator='\n')
    writer.writerow(attrs)

    def write_log(fetch, **kwargs):
        for (key, label, logit) in zip(*[fetch[attr] for attr in attrs]):
            writer.writerow([key, label, logit])

    model.test(
        iteration=test_iteration,
        feed_dict={phase: TEST},
        callbacks=[
            dict(fetch=display_dict[TEST],
                 func=lambda **kwargs: model.display(begin='\033[2K\rTest', end='', **kwargs)),
            dict(interval=-1,
                 func=lambda **kwargs: print('')),
            dict(fetch={attr: globals()[attr] for attr in attrs},
                 func=write_log)])

    log_handle.close()

if __name__ == '__main__':
    if FLAGS.command == 'train':
        train()
    elif FLAGS.command == 'test':
        test()
    elif FLAGS.command == 'none':
        pass
    else:
        assert False, 'Bad command'
