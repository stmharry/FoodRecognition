#!/usr/bin/env python
from __future__ import print_function

import gflags
import numpy as np
import os
import shutil
import sys

import data
import log_reader

gflags.DEFINE_string('log_file', '/tmp/test_log.csv', None)
gflags.DEFINE_string('train_dir', None, None)
gflags.DEFINE_string('image_dir', '/tmp/incorrect', None)
FLAGS = gflags.FLAGS
FLAGS(sys.argv)

data.cache_files(directory=FLAGS.train_dir)
(keys, labels, logits) = log_reader.read(FLAGS.log_file)

if os.path.isdir(FLAGS.image_dir):
    shutil.rmtree(FLAGS.image_dir)
os.makedirs(FLAGS.image_dir)

for (key, label, logit) in zip(keys, labels, logits):
    prediction = np.argmax(logit)
    if label != logit:
        print('%s' % key)
        (dirname, basename) = os.path.split(key)

        if label == -1:
            format_str = '%s(%.2f)_%s'
            format_tuple = (data.CLASS_NAMES[prediction], logit[prediction], basename)
        else:
            format_str = '%s->%s(%.2f)_%s'
            format_tuple = (data.CLASS_NAMES[label], data.CLASS_NAMES[prediction], logit[prediction], basename)

        new_key = os.path.join(FLAGS.image_dir, format_str % format_tuple)
        shutil.copyfile(key, new_key)
