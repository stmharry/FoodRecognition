#!/usr/bin/env python
from __future__ import print_function

import gflags
import numpy as np
import pylab
import sys

import data
import log_reader

gflags.DEFINE_string('log_file', '/tmp/test_log.csv', None)
gflags.DEFINE_string('train_dir', None, None)
gflags.DEFINE_string('confusion_mat_file', 'confusion_mat.pdf', None)
gflags.DEFINE_string('accs_file', 'accs.pdf', None)
FLAGS = gflags.FLAGS
FLAGS(sys.argv)

data.cache_files(directory=FLAGS.train_dir)
num_classes = len(data.CLASS_NAMES)
range_classes = np.arange(num_classes)
(keys, labels, logits) = log_reader.read(FLAGS.log_file)

confusion_mat = np.zeros((num_classes, num_classes))
for (key, label, logit) in zip(keys, labels, logits):
    prediction = np.argmax(logit)
    confusion_mat[prediction][label] += 1

confusion_mat_norm = confusion_mat / np.sum(confusion_mat, axis=0)
confusion_mat_imshow = np.power(confusion_mat_norm, 0.3)
confusion_mat_annotate = np.round(1000 * confusion_mat_norm)

pylab.figure(hash('confusion_mat'))
pylab.imshow(confusion_mat_imshow, interpolation='nearest', cmap='GnBu')
pylab.setp(pylab.gca().get_xticklines(), visible=False)
pylab.setp(pylab.gca().get_yticklines(), visible=False)

for x in range_classes:
    for y in range_classes:
        pylab.annotate(
            '%d' % confusion_mat_annotate[y, x],
            xy=(x, y),
            horizontalalignment='center',
            verticalalignment='center',
            size='xx-small')

pylab.xlabel('Ground Truth', size='larger')
pylab.ylabel('Prediction', size='larger')
pylab.xticks(range_classes, data.CLASS_NAMES, rotation='vertical')
pylab.yticks(range_classes, data.CLASS_NAMES)
pylab.tight_layout()

pylab.savefig(FLAGS.confusion_mat_file)

#
accs = np.diag(confusion_mat) / np.sum(confusion_mat, axis=0)
acc = np.sum(np.diag(confusion_mat)) / np.sum(confusion_mat)

pylab.figure(hash('accs'))
pylab.bar(range_classes, accs, color='0.75')
pylab.plot([0, num_classes], [acc, acc], 'k:')
pylab.annotate(
    'Overall: %.3f' % acc,
    xy=(num_classes / 2., acc),
    horizontalalignment='center',
    verticalalignment='bottom')
pylab.setp(pylab.gca().get_xticklines(), visible=False)
pylab.setp(pylab.gca().get_yticklines(), visible=False)

pylab.xlim([0, num_classes])
pylab.ylim([0.5, 1.0])
pylab.ylabel('Accuracy', size='larger')
pylab.xticks(range_classes + 0.4, data.CLASS_NAMES, rotation='vertical', size='large')
pylab.tight_layout()

pylab.savefig(FLAGS.accs_file)
