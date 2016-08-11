#!/usr/bin/env python

import caffe
import numpy as np
import scipy.io

DIR = 'archive/'

NET_PROTO = DIR + 'ResNet-50-deploy.prototxt'
NET_MODEL = DIR + 'ResNet-50-model.caffemodel'
NET_MEAN = DIR + 'ResNet_mean.binaryproto'

PARAMS_PATH = DIR + 'ResNet-50-params.mat'
MEAN_PATH = DIR + 'ResNet-mean.mat'

caffe.set_mode_cpu()
resnet = caffe.Net(NET_PROTO, NET_MODEL, caffe.TRAIN)

mdict = dict()
for (key, value) in resnet.params.iteritems():
    print key
    mdict[key] = np.zeros((len(value),), dtype=np.object)
    for l in xrange(len(value)):
        print value[l].data.shape
        if value[l].data.ndim == 4:
            mdict[key][l] = np.transpose(value[l].data, axes=(2, 3, 1, 0))
        else:
            mdict[key][l] = value[l].data

scipy.io.savemat(PARAMS_PATH, mdict, do_compression=True, oned_as='column')

mean_file = open(NET_MEAN, mode='rb')
mean_data = mean_file.read()
mean_blob = caffe.proto.caffe_pb2.BlobProto()
mean_blob.ParseFromString(mean_data)
mean = caffe.io.blobproto_to_array(mean_blob)[0].transpose((1, 2, 0)).astype(np.float32)

scipy.io.savemat(MEAN_PATH, dict(mean=mean), do_compression=True, oned_as='column')
