import time

from ResNet import Meta, InputProducer, ImageLabelPipeline, Batch, Net, ResNet50

WORKING_DIR = '/mnt/data/dish-clean-save/' + time.strftime('%Y-%m-%d-%H%M%S')
IMAGE_DIR = '/mnt/data/dish-clean/'
SUBSAMPLE_SIZE = 64  # train:val = 63:1
IS_IMAGE_CHECKED = True

Meta.load(working_dir=WORKING_DIR)

inputProducer = InputProducer()
imageLabelPipeline = ImageLabelPipeline()
batch = Batch()

trainImageLabel = inputProducer.fromFile(
    image_dir=IMAGE_DIR,
    is_train=True,
    subsample_size=SUBSAMPLE_SIZE,
    subsample_divisible=False,
    check=not IS_IMAGE_CHECKED,
    shuffle=True)
trainImageLabel = imageLabelPipeline.train(*trainImageLabel)
trainImageLabel = batch.train(*trainImageLabel)

testImageLabel = inputProducer.fromFile(
    image_dir=IMAGE_DIR,
    subsample_size=SUBSAMPLE_SIZE)
testImageLabel = imageLabelPipeline.test(*testImageLabel)
testImageLabel = batch.test(*testImageLabel)

net = ResNet50(
    learning_rate=1e-1,
    learning_rate_decay_steps=1500,
    learning_rate_decay_rate=0.5,
    is_train=True,
    is_show=True)

(image, label) = net.case([
    (Net.Phase.TRAIN, lambda: trainImageLabel),
    (Net.Phase.TEST, lambda: testImageLabel)])
image.set_shape((batch.batch_size,) + imageLabelPipeline.shape)
label.set_shape((None,))

net.build(image=image, label=label)
net.train(iteration=10000)
