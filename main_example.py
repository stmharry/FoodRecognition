from ResNet import Meta, InputProducer, ImagePipeline, Net, ResNet50

Meta.load(working_dir='/tmp')

inputImage = Net.placeholder('image')
(image, label) = InputProducer().fromPlaceholder(image=inputImage)
image = ImagePipeline().online(image)
net = ResNet50(image=image, label=label)


def get_prob(X):
    fetch_val = net.online(
        feed_dict={inputImage: X},
        fetch={'prob': net.prob})
    return fetch_val['prob']
