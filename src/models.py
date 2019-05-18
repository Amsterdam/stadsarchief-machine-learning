from keras.applications.vgg16 import VGG16
from keras import layers, initializers
from keras.models import Model, Sequential


def build_vgg16(num_classes, img_dim):
    # see https://keras.io/applications/#vgg16
    assert(img_dim[0] == 224)
    assert(img_dim[1] == 224)
    assert(img_dim[2] == 3)

    vgg16 = VGG16(
        include_top=False,
        input_shape=img_dim,
        weights='imagenet',  # pre-trained on ImageNet
    )

    x = layers.Flatten(name='flatten')(vgg16.layers[-2].output)
    x = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

    model = Model(input=vgg16.input, output=x)

    return model


lr_alpha=0.1
drop_chance=0.0


def create_mlp(num_features, num_classes=None):
    model = Sequential()
    model.add(layers.Dense(64, input_dim=num_features, name="input-mlp"))
    #     model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=lr_alpha))
    model.add(layers.Dropout(drop_chance))

    model.add(layers.Dense(64))
    #     model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=lr_alpha))
    model.add(layers.Dropout(drop_chance))

    if num_classes is not None:
        model.add(layers.Dense(2, activation='softmax'))

    return model


def create_cnn(img_dim, num_classes=None):
    inputs = layers.Input(shape=img_dim, name="input-cnn")

    x = layers.Conv2D(16, (3, 3), use_bias=False)(inputs)
    #     x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=lr_alpha)(x)

    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(32, (3, 3), use_bias=False)(x)
    #     x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=lr_alpha)(x)

    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(32, (3, 3), use_bias=False)(x)
    #     x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=lr_alpha)(x)

    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(32, (3, 3), use_bias=False)(x)
    #     x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=lr_alpha)(x)

    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Flatten()(x)

    x = layers.Dense(16, kernel_initializer = initializers.RandomUniform())(x)
    #     x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=lr_alpha)(x)
    x = layers.Dropout(drop_chance)(x)

    if num_classes is not None:
        x = layers.Dense(2, activation="softmax", name='output')(x)

    model = Model(inputs, x)
    return model


def create_cnn_deep(img_dim, num_classes=None):
    inputs = layers.Input(shape=img_dim, name="input-cnn")

    # Block 1
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1')(inputs)
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1')(x)
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    x = layers.Flatten()(x)

    x = layers.Dense(8, kernel_initializer=initializers.RandomUniform())(x)
    x = layers.LeakyReLU(alpha=lr_alpha)(x)
    # x = layers.Dropout(drop_chance)(x)

    x = layers.Dense(8, kernel_initializer=initializers.RandomUniform())(x)
    x = layers.LeakyReLU(alpha=lr_alpha)(x)
    # x = layers.Dropout(drop_chance)(x)

    if num_classes is not None:
        x = layers.Dense(2, activation="softmax", name='output')(x)

    model = Model(inputs, x)
    return model


def build_multi_feature(num_classes, img_dim, num_features):
    # Multi feature system based on https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/
    print('img_dim: ', img_dim)
    print('num_features: ', num_features)

    # create the MLP and CNN models
    mlp = create_mlp(num_features)
    cnn = create_cnn(img_dim)

    # Merge the two branches
    combinedInput = layers.concatenate([mlp.output, cnn.output])

    # Final logic
    x = layers.Dense(8, activation="relu")(combinedInput)
    x = layers.Dropout(drop_chance)(x)

    x = layers.Dense(num_classes, activation="softmax", name='output')(x)

    # our final model will accept categorical/numerical data on the MLP
    # input and images on the CNN input
    # Output: softmax
    model = Model(inputs=[cnn.input, mlp.input], outputs=x)
    return model
