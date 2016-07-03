#!/usr/bin/env python
import csv
import gflags
import numpy as np
import os
import pylab
import re
import shutil
import sys

import data

gflags.DEFINE_bool('make_stats', False, 'whether make statistics of outputs.')
gflags.DEFINE_bool('make_images', False, 'whether collect wrongly classified images.')
gflags.DEFINE_string('log_file', '/tmp/test_log.csv', 'log file to use.')
gflags.DEFINE_string('confusion_mat_file', 'confusion_mat.pdf', 'output for confusion matrix.')
gflags.DEFINE_string('accs_file', 'accs.pdf', 'output for accuracy.')
gflags.DEFINE_string('train_dir', None, 'directory for files to be trained.')
gflags.DEFINE_string('image_dir', '/tmp/log_images', 'directory for logging images.')
FLAGS = gflags.FLAGS
FLAGS(sys.argv)


def make_stats(keys, labels, logits):
    confusion_mat = np.zeros((num_classes, num_classes))
    for (key, label, logit) in zip(keys, labels, logits):
        confusion_mat[np.argmax(logit)][label] += 1

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
    pylab.xticks(range_classes, class_names, rotation='vertical')
    pylab.yticks(range_classes, class_names)
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
    pylab.xticks(range_classes + 0.4, class_names, rotation='vertical', size='large')
    pylab.tight_layout()

    pylab.savefig(FLAGS.accs_file)


def make_images(values):
    if os.path.isdir(FLAGS.image_dir):
        shutil.rmtree(FLAGS.image_dir)
    os.makedirs(FLAGS.image_dir)

    for (key, label, logit) in zip(keys, labels, logits):
        prediction = np.argmax(logit)
        if label != logit:
            (dirname, basename) = os.path.split(key)

            if label == -1:
                format_str = '%s(%.2f)_%s'
                format_tuple = (data.CLASS_NAMES[prediction], logit[prediction], basename)
            else:
                format_str = '%s->%s(%.2f)_%s'
                format_tuple = (data.CLASS_NAMES[label], data.CLASS_NAMES[prediction], logit[prediction], basename)

            new_key = os.path.join(FLAGS.image_dir, format_str % format_tuple)
            shutil.copyfile(key, new_key)
            print('%s\n-> %s' % (key, new_key))


data.cache_files(image_dir=FLAGS.train_dir)
class_names = data.CLASS_NAMES
num_classes = len(data.CLASS_NAMES)
range_classes = np.arange(num_classes)

log_handle = open(FLAGS.log_file, 'r')
reader = csv.reader(log_handle)
reader.next()

values = []
for row in reader:
    (key, label, logit) = (
        row[0],
        int(row[1]),
        np.fromstring(re.sub('[\[\]]', '', row[2].replace('\n', '')), sep=' '))
    values.append((key, label, logit))
log_handle.close()

(keys, labels, logits) = [np.stack(value, axis=0) for value in zip(*values)]

if FLAGS.make_stats:
    make_stats(keys, labels, logits)

if FLAGS.make_images:
    make_images(keys, labels, logits)
