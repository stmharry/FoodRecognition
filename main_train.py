import time
from ResNet import Meta, Blob, Producer, Preprocess, Batch, Net, ResNet50

WORKING_DIR = '/mnt/data/dish-clean-save/' + time.strftime('%Y-%m-%d-%H%M%S')
IMAGE_DIR = '/mnt/data/dish-clean/'
IS_IMAGE_CHECKED = True

Meta.load(working_dir=WORKING_DIR)
producer = Producer()
preprocess = Preprocess()
batch = Batch()

trainBlob = producer.trainFile(image_dir=IMAGE_DIR, check=not IS_IMAGE_CHECKED).func(preprocess.train).func(batch.train)
testBlob = producer.testFile(image_dir=IMAGE_DIR).func(preprocess.test).func(batch.test)

net = ResNet50(
    learning_rate=1e-1,
    learning_rate_decay_steps=1500,
    learning_rate_decay_rate=0.5,
    is_train=True,
    is_show=True)

image = net.case([
    (Net.Phase.TRAIN, lambda: trainBlob.image),
    (Net.Phase.TEST, lambda: testBlob.image)],
    shape=(batch.batch_size,) + preprocess.shape)
label = net.case([
    (Net.Phase.TRAIN, lambda: trainBlob.label),
    (Net.Phase.TEST, lambda: testBlob.label)],
    shape=(None,))
blob = Blob(image=image, label=label)

net.build(blob)
net.train(iteration=10000)
