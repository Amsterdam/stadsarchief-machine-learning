from keras.applications.vgg16 import VGG16
from keras.layers import Flatten, Dense
from keras.models import Model


def build_vgg16(num_classes, img_dim):
    # see https://keras.io/applications/#vgg16
    assert(img_dim[0] == 224)
    assert(img_dim[1] == 224)
    assert(img_dim[2] == 3)

    vgg16 = VGG16(
        include_top=False,
        input_shape=img_dim,
        weights='imagenet', # pre-trained on ImageNet
    )

    x = Flatten(name='flatten')(vgg16.layers[-2].output)
    x = Dense(num_classes, activation='softmax', name='predictions')(x)

    model = Model(input=vgg16.input, output=x)

    return model

