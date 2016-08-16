import cStringIO
import numpy as np
import pandas
import PIL.Image
import urllib

from ResNet import Meta, Blob, Preprocess, Net, ResNet50

WORKING_DIR = '/mnt/data/dish-clean-save/2016-08-16-191753/'

Meta.load(working_dir=WORKING_DIR)
preprocess = Preprocess()

image = Net.placeholder('image')
blob = Blob(image=image).func(preprocess.test)

net = ResNet50()
net.build(blob)


def get_prob(image_val):
    fetch_val = net.online(
        feed_dict={image: image_val},
        fetch={'prob': net.prob})
    return fetch_val['prob']


def get_image_from_url(url):
    pipe = urllib.urlopen(url)
    stringIO = cStringIO.StringIO(pipe.read())
    pil = PIL.Image.open(stringIO)
    image_val = np.array(pil.getdata(), dtype=np.uint8).reshape((pil.height, pil.width, -1))
    return image_val


def get_prob_from_url(url):
    image_val = get_image_from_url(url)
    prob = get_prob(image_val)
    df = pandas.DataFrame(data=prob, columns=Meta.CLASS_NAMES)
    return df
