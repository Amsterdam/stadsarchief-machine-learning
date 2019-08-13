# coding: utf-8

import os
import shutil
import time

import keras
import keras_metrics
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.python.client import device_lib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, recall_score, accuracy_score


from src.stats import show_prediction_list, show_prediction_images
from src.processing.TargetEncoder import TargetEncoder
from src.processing.ImageFeatureEncoder import ImageFeatureEncoder


def tensorflow_gpu_supported():
    return {
        'gpus': K.tensorflow_backend._get_available_gpus(),
        'devices': device_lib.list_local_devices()
    }


def gpu_memory_to_on_demand():
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


def split_bool_arrays(predictions: np.ndarray, threshold, verbose=False):
    assert predictions.shape[1] == 2, 'expecting binary prediction in one hot format'

    y_pred_conf = np.amax(predictions, axis=1)

    if verbose:
        print(y_pred_conf.round(2)[:10])

    certain = y_pred_conf >= threshold
    uncertain = np.invert(certain)
    return [certain, uncertain]


def split_uncertain(predictions: np.ndarray, threshold, elements, verbose=False):
    """
    Split all elements into certain and uncertain buckets

    @return [[elem1_certain, elem1_uncertain], ...]
    """
    for element in elements:
        assert element.shape[0] == predictions.shape[0], 'number of an element not equal to number of predictions'

    [certain, uncertain] = split_bool_arrays(predictions, threshold, verbose=verbose)

    results = []
    for element in elements:
        certain_bucket = element[certain]
        uncertain_bucket = element[uncertain]
        results.append([certain_bucket, uncertain_bucket])
    return results


def print_missing_types_error(unique_types: list, expected: list, name: str):
    print(f'{name} samples do not reflect all types, recall, precision and T1 scores therefor do not make much sense.')
    print(f'{name} types: {unique_types}, expected {expected}')
    print('Classification report generation skipped.')


def create_reports(y_true_oh: np.ndarray, y_pred_oh: np.ndarray, labelEncoder, labels):
    assert y_true_oh.shape == y_pred_oh.shape
    assert y_true_oh.shape[1] == 2, 'expecting binary one hot inputs'
    y_true = labelEncoder.inverse_transform(y_true_oh)
    y_pred = labelEncoder.inverse_transform(y_pred_oh)

    is_correct=y_true == y_pred

    y_true_pd = pd.Series(y_true.ravel())
    y_pred_pd = pd.Series(y_pred.ravel())
    crosstab = pd.crosstab(y_true_pd, y_pred_pd, rownames=['Actual'], colnames=['Predicted'], margins=True, margins_name='Total')

    report = None
    true_types = np.unique(y_true_pd)
    pred_types = np.unique(y_pred_pd)
    if len(true_types) < len(labels):
        print_missing_types_error(true_types, labels, 'True')
    elif len(pred_types) < len(labels):
        print_missing_types_error(pred_types, labels, 'Pred')
    else:
        report = classification_report(y_true, y_pred, labels=labels)

    return [crosstab, report, is_correct]


def show_reports(y_true_oh: np.ndarray, y_pred_oh: np.ndarray, labelEncoder, labels):
    if len(y_true_oh) == 0 or len(y_pred_oh) == 0:
        print("No results to show")
        return
    [crosstab, report, is_correct] = create_reports(y_true_oh, y_pred_oh, labelEncoder, labels)
    print(crosstab)

    if (report):
        print()
        print(report)

    return is_correct



def predictions_overview(Y_oh, pred_oh, references, encoder):
    Y_class = encoder.inverse_transform(Y_oh)
    pred_class = encoder.inverse_transform(pred_oh)

    data = {'reference': references, 'label': Y_class[:, 0], 'prediction': pred_class[:, 0]}
    df = pd.DataFrame(data)

    return df


def write_model(model, directory):
    model_json = model.to_json()
    json_path = os.path.join(directory, "model.json")
    weights_path = os.path.join(directory, "weights.h5")
    with open(json_path, "w") as json_file:
        json_file.write(model_json)
    model.save_weights(weights_path)
