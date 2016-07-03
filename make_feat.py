#!/usr/bin/env python
from __future__ import print_function

import csv
import gflags
import joblib
import numpy as np
import PIL
import pylab
import re
import sys

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import BallTree

gflags.DEFINE_bool('make_tsne', False, 'whether to make TSNE.')
gflags.DEFINE_bool('make_tree', False, 'whether to make KD-tree.')

gflags.DEFINE_string('log_file', '/tmp/test_log.csv', 'log file to use.')
gflags.DEFINE_string('tsne_image_file', 'tsne.png', 'output image for tsne.')
gflags.DEFINE_string('tsne_scatter_file', 'tsne.pdf', 'output scatter plot for tsne.')
gflags.DEFINE_float('tsne_sample_ratio', 1.0, 'ratio of files to show.')
gflags.DEFINE_integer('tsne_canvas_size', 4096, 'size for the canvas.')
gflags.DEFINE_integer('tsne_max_size', 48, 'max size for a single image.')
gflags.DEFINE_string('tree_file', 'tree', 'dumped tree.')
FLAGS = gflags.FLAGS
FLAGS(sys.argv)


def make_tsne(keys, labels, feats):
    pca = PCA(n_components=128)
    tsne = TSNE(init='pca', verbose=2)
    feats = tsne.fit_transform(pca.fit_transform(feats))

    (x_min, y_min) = np.min(feats, axis=0)
    (x_max, y_max) = np.max(feats, axis=0)

    x_range = x_max - x_min
    y_range = y_max - y_min
    if x_range > y_range:
        x_res = int(np.ceil(x_range / y_range * FLAGS.tsne_canvas_size))
        y_res = FLAGS.tsne_canvas_size
    else:
        x_res = FLAGS.tsne_canvas_size
        y_res = int(np.ceil(y_range / x_range * FLAGS.tsne_canvas_size))

    canvas = np.ones((y_res + FLAGS.tsne_max_size, x_res + FLAGS.tsne_max_size, 3), dtype=np.uint8) * 255
    for ((x, y), key) in zip(feats, keys):
        if not np.random.random_sample() < FLAGS.tsne_sample_ratio:
            continue

        print(key)
        image = PIL.Image.open(key)
        (width, height) = map(float, image.size)
        if width < height:
            size = (int(width / height * FLAGS.tsne_max_size), FLAGS.tsne_max_size)
        else:
            size = (FLAGS.tsne_max_size, int(height / width * FLAGS.tsne_max_size))
        image = image.resize(size, resample=PIL.Image.BILINEAR)

        x_index = int(np.round((x - x_min) / x_range * x_res))
        y_index = int(np.round((y - y_min) / y_range * y_res))
        canvas[y_index:y_index + image.height, x_index:x_index + image.width] = image

    PIL.Image.fromarray(canvas).save(FLAGS.tsne_image_file)

    pylab.scatter(feats[:, 0], feats[:, 1], s=5, c=labels, marker='o', linewidth=0)
    pylab.savefig(FLAGS.tsne_scatter_file)


def make_tree(keys, labels, feats):
    tree = BallTree(feats)
    joblib.dump(tree, FLAGS.tree_file, compress=3)


log_handle = open(FLAGS.log_file, 'r')
reader = csv.reader(log_handle)
reader.next()

values = []
for (num_row, row) in enumerate(reader):
    (key, label, feat) = (
        row[0],
        int(row[1]),
        np.fromstring(re.sub('[\[\]]', '', row[2].replace('\n', '')), sep=' '))
    values.append((key, label, feat))
    print('\033[2K\r%d: %s (%s)' % (num_row, key, label), end='')
    sys.stdout.flush()
log_handle.close()
print('')

(keys, labels, feats) = [np.stack(value, axis=0) for value in zip(*values)]

if FLAGS.make_tsne:
    make_tsne(keys, labels, feats)

if FLAGS.make_tree:
    make_tree(keys, labels, feats)
