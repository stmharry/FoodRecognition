#!/usr/bin/env python
from __future__ import print_function

import gflags
import numpy as np
import PIL
import pylab
import sys

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import log_reader

gflags.DEFINE_string('log_file', '/tmp/test_log.csv', None)
gflags.DEFINE_string('tsne_file', 'tsne.png', None)
gflags.DEFINE_string('scatter_file', 'tsne.pdf', None)
gflags.DEFINE_float('sample_ratio', 1.0, None)
gflags.DEFINE_integer('canvas_size', 4096, None)
gflags.DEFINE_integer('image_size', 48, None)
FLAGS = gflags.FLAGS
FLAGS(sys.argv)

(keys, labels, feats) = log_reader.read(FLAGS.log_file)

pca = PCA(n_components=128)
tsne = TSNE(init='pca', verbose=2)
feats = tsne.fit_transform(pca.fit_transform(feats))

(x_min, y_min) = np.min(feats, axis=0)
(x_max, y_max) = np.max(feats, axis=0)

x_range = x_max - x_min
y_range = y_max - y_min
if x_range > y_range:
    x_res = int(np.ceil(x_range / y_range * FLAGS.canvas_size))
    y_res = FLAGS.canvas_size
else:
    x_res = FLAGS.canvas_size
    y_res = int(np.ceil(y_range / x_range * FLAGS.canvas_size))

canvas = np.ones((y_res + FLAGS.image_size, x_res + FLAGS.image_size, 3), dtype=np.uint8) * 255
for ((x, y), key) in zip(feats, keys):
    if not np.random.random_sample() < FLAGS.sample_ratio:
        continue

    print(key)
    image = PIL.Image.open(key)
    (width, height) = map(float, image.size)
    if width < height:
        size = (int(width / height * FLAGS.image_size), FLAGS.image_size)
    else:
        size = (FLAGS.image_size, int(height / width * FLAGS.image_size))
    image = image.resize(size, resample=PIL.Image.BILINEAR)

    x_index = int(np.round((x - x_min) / x_range * x_res))
    y_index = int(np.round((y - y_min) / y_range * y_res))
    canvas[y_index:y_index + image.height, x_index:x_index + image.width] = image

PIL.Image.fromarray(canvas).save(FLAGS.tsne_file)

pylab.scatter(feats[:, 0], feats[:, 1], s=5, c=labels, marker='o', linewidth=0)
pylab.savefig(FLAGS.scatter_file)
