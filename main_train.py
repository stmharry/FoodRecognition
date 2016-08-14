from ResNet import Input, ImagePipeline, Net, ResNet50

/working_dir = '/tmp'

image = Net.placeholder(name='image')
imagePipeline = ImagePipeline()
net = ResNet50(image=imagePipeline.online(image), class_name=None)


def get_prob(X):
    fetch_val = net.online(
        feed_dict={image: X},
        fetch={'prob': net.prob})
    return fetch_val['prob']
