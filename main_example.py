from ResNet import ImagePipeline, Net, ResNet50


image = Net.placeholder(name='image')
imagePipeline = ImagePipeline()

net = ResNet50(image=imagePipeline.online(image), class_names=)


def get_prob(X):
    fetch_val = net.online(
        feed_dict={image: X},
        fetch={'prob': net.prob})
    return fetch_val['prob']
