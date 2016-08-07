#!/usr/bin/env python
from __future__ import print_function

import gflags
import numpy as np
import os
import PIL
import pylab
import shutil
import sys
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.neighbors import BallTree

import log_reader

gflags.DEFINE_string('train_log_file', None, None)
gflags.DEFINE_string('test_log_file', None, None)
gflags.DEFINE_string('image_dir', '/tmp/tree', None)
gflags.DEFINE_integer('top_k', 5, None)
FLAGS = gflags.FLAGS
FLAGS(sys.argv)

(train_keys, train_labels, train_feats) = log_reader.read(FLAGS.train_log_file)
train_feats /= (np.sqrt(np.sum(np.square(train_feats), axis=1, keepdims=True)) * 2)
(test_keys, test_labels, test_feats) = log_reader.read(FLAGS.test_log_file)
test_feats /= (np.sqrt(np.sum(np.square(test_feats), axis=1, keepdims=True)) * 2)

if os.path.isdir(FLAGS.image_dir):
    shutil.rmtree(FLAGS.image_dir)
os.makedirs(FLAGS.image_dir)

start_time = time.time()
tree = BallTree(train_feats)
print('Build time: %.2fs' % (time.time() - start_time))

for (num, (key, label, feat)) in enumerate(zip(test_keys, test_labels, test_feats)):
    (dirname, basename) = os.path.split(key.item())
    (name, ext) = os.path.splitext(basename)

    (dist, ind) = tree.query(feat, k=FLAGS.top_k)
    (fig, axs) = pylab.subplots(
        ncols=1 + FLAGS.top_k,
        squeeze=False,
        figsize=(3 * (1 + FLAGS.top_k), 4))

    keys = [key] + [train_keys[i] for i in ind[0, :]]
    titles = ['Query'] + ['dist=%.4f' % d for d in dist[0, :]]

    for (ax, key, title) in zip(axs[0, :], keys, titles):
        image = PIL.Image.open(key.item())
        ax.imshow(np.asarray(image))
        ax.set_title(title)
        ax.axis('off')

    new_name = os.path.join(FLAGS.image_dir, '%s.png' % name)
    pylab.savefig(new_name)
    print('\033[2K\rMaking query summary #%d: %s' % (num, new_name), end='')

print('')
