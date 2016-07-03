#!/usr/bin/env python
from __future__ import print_function

import gflags
import sys
import tensorflow as tf

import data

gflags.DEFINE_string('train_dir', None, '')
gflags.DEFINE_integer('batch_size', 64, '')
FLAGS = gflags.FLAGS
FLAGS(sys.argv)

data.cache_train_files(train_dir=FLAGS.train_dir, recache=True)
train_files = data.get_files(data.TRAIN, num_pipelines=8)
values = data.get_test_values(train_files, batch_size=FLAGS.batch_size, num_test_crops=1)

sess = tf.Session()
tf.train.start_queue_runners(sess=sess)

num_batches = int(sum(map(len, train_files)) / FLAGS.batch_size) + 1
for i in xrange(num_batches):
    print('%d / %d\r' % (i, num_batches), end='')
    sys.stdout.flush()
    sess.run(values)
