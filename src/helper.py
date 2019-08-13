# coding: utf-8

import os
from keras import backend as K
import tensorflow as tf
from tensorflow.python.client import device_lib
import numpy as np


from src.processing.TargetEncoder import TargetEncoder
from src.processing.ImageFeatureEncoder import ImageFeatureEncoder


def tensorflow_gpu_supported():
    return {
        'gpus': K.tensorflow_backend._get_available_gpus(),
        'devices': device_lib.list_local_devices()
    }


def gpu_memory_to_on_demand():
    """
    Set tensorflow GPU memory usage to on demand rather than preallocate.
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    K.tensorflow_backend.set_session(tf.Session(config=config))


def create_model_dirs(model_dir):
    transform_dir = os.path.join(model_dir, 'transform/')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(transform_dir, exist_ok=True)
    return model_dir, transform_dir


# def print_shapes():
#     things = ('Xtrain', 'Ytrain', 'Ztrain', 'Xvalid', 'Yvalid', 'Zvalid')
#     for thing in things:
#         print(f"shape {thing}: ")


def feature_preprocess(Xtrain_raw, Xvalid_raw):
    imageEncoder = ImageFeatureEncoder()

    imageEncoder.fit(Xtrain_raw)
    print('fitted encoder to training set')

    Xtrain = imageEncoder.transform(Xtrain_raw)
    Xvalid = imageEncoder.transform(Xvalid_raw)
    print('Xtrain.shape: ', Xtrain.shape)
    print('Xvalid.shape: ', Xvalid.shape)

    return Xtrain, Xvalid, imageEncoder


def label_preprocessing(Ztrain_raw, Zvalid_raw, num_classes=2):
    classes = list(set(Ztrain_raw.ravel()))
    assert len(classes) == num_classes, f'{len(classes)} classes in training set: {classes}, expected: {num_classes}'

    labelEncoder = TargetEncoder()

    labelEncoder.fit(Ztrain_raw)

    Ztrain = labelEncoder.transform(Ztrain_raw).toarray()
    Zvalid = labelEncoder.transform(Zvalid_raw).toarray()

    labels = np.array([['aanvraag'], ['other']])

    return classes, Ztrain, Zvalid, labelEncoder, labels


