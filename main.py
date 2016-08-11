from ResNet import Image, Net, ResNet50

phase = Net.placeholder(name='phase')
image = Net.placeholder(name='image', shape=None)
label = Net.placeholder(name='label')

net = ResNet50(
    phase=phase,
    image=Image.online(image),
    label=label)


def logit(X):
    net.test(
        feed_dict={
            phase: Net.Phase.TEST,  # Always Net.Phase.TEST
            image: X,  # Image of arbitrary size, with values between 0 and 255
            label: -1},  # (-1) denotes unknown label
        fetch={
            'logit': net.logit})
    return net.output_values['logit']
