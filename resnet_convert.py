import caffe
import numpy as np
import scipy.io

NET_PROTO = '/home/user/Desktop/ResNet-50-deploy.prototxt'
NET_MODEL = '/home/user/Desktop/ResNet-50-model.caffemodel'
NET_MEAN = '/home/user/Desktop/ResNet_mean.binaryproto'

MAT_PATH = '/home/user/Desktop/ResNet-50-params.mat'

caffe.set_mode_cpu()
resnet = caffe.Net(NET_PROTO, NET_MODEL, caffe.TRAIN)

mdict = {}
for (key, value) in resnet.params.iteritems():
    print key
    mdict[key] = np.zeros((len(value),), dtype=np.object)
    for l in xrange(len(value)):
        print value[l].data.shape
        if value[l].data.ndim == 4:
            mdict[key][l] = np.transpose(value[l].data, axes=(2, 3, 1, 0))
        else:
            mdict[key][l] = value[l].data

mean_file = open(NET_MEAN, mode='rb')
mean_data = mean_file.read()
mean_blob = caffe.proto.caffe_pb2.BlobProto()
mean_blob.ParseFromString(mean_data)
mean = caffe.io.blobproto_to_array(mean_blob)[0].transpose((1, 2, 0)).astype(np.float32)
mdict.update({'mean': mean})

scipy.io.savemat(MAT_PATH, mdict, do_compression=True, oned_as='column')
