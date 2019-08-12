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


from src.stats import list_stats, show_train_curves, show_prediction_list, show_prediction_images
from src.processing.TargetEncoder import TargetEncoder
from src.processing.ImageFeatureEncoder import ImageFeatureEncoder
from src import models as own_models


class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


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


def preprocess(Xtrain_raw, Xvalid_raw):
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

    print('\n--- Train ---')
    list_stats(Ztrain_raw.ravel())

    print('\n--- Valid ---')
    list_stats(Zvalid_raw.ravel())

    labelEncoder = TargetEncoder()

    labelEncoder.fit(Ztrain_raw)

    Ztrain = labelEncoder.transform(Ztrain_raw).toarray()
    Zvalid = labelEncoder.transform(Zvalid_raw).toarray()

    labels = np.array([['aanvraag'], ['other']])

    print()
    print('--- transform ---')
    print(f'labels:\n{labels}')
    print(f'encoded:\n{labelEncoder.transform(labels).toarray()}')

    print()
    print('--- inverse transform ---')
    example = [[1, 0.5]]
    print(f'inverse transform for example {example}: {labelEncoder.inverse_transform(example)}')

    print()
    print('--- transform on subset of data ---')
    print_count = 10  # examples to show
    print(f'Ztrain_raw[:{print_count}]:\n{Ztrain_raw[:print_count]}')
    print('Ztrain:\n', Ztrain[:print_count])

    print()
    print('--- transformation shapes ---')
    print('Ztrain_raw shape:\t', Ztrain_raw.shape)
    print('Ztrain shape:\t\t', Ztrain.shape)
    print('Zvalid_raw shape:\t', Ztrain_raw.shape)
    print('Zvalid shape:\t\t', Zvalid.shape)

    return classes, Ztrain, Zvalid, labelEncoder, labels


def define_model(img_dim, num_classes=2):
    # model = own_models.create_cnn(img_dim, num_classes=num_classes)
    model = own_models.create_cnn_g(img_dim, num_classes=num_classes, drop_chance=0.6)
    model.summary()
    return model


def is_binary(model):
    n_classes = model.get_layer('output').output_shape[1]
    return n_classes == 1


def compile_model(model):
    assert(K.image_data_format() == 'channels_last')

    #     if is_binary(model):
    #         loss= keras.losses.binary_crossentropy
    #     else:
    loss=keras.losses.categorical_crossentropy

    model.compile(
        loss=loss,
        #         optimizer=keras.optimizers.Adadelta(),
        #         optimizer='rmsprop',
        #         optimizer='sgd',
        #         optimizer=keras.optimizers.SGD(lr=0.01),
        #         optimizer=keras.optimizers.Adam(),
        optimizer=keras.optimizers.Adam(lr=0.0003),
        #         optimizer=keras.optimizers.Adam(lr=0.0001),
        #         optimizer=keras.optimizers.Adam(lr=0.00003),
        #         metrics=['accuracy', keras_metrics.recall()]
        metrics=['accuracy', keras_metrics.binary_recall(label=0)]

    )


def train_gen(model, datagen, class_weight, X_train, Z_train, X_valid, Z_valid, batch_size, epochs, tbCallBack, terminateCB):
    compile_model(model)

    history = model.fit_generator(
        datagen.flow(X_train,
                     Z_train,
                     batch_size=batch_size
                     ),
        steps_per_epoch=int(np.ceil(X_train.shape[0] / float(batch_size))),
        epochs=epochs,
        validation_data=(X_valid, Z_valid),
        class_weight=class_weight,
        workers=4,
        callbacks=[tbCallBack, terminateCB]
    )
    return history


def train(model, classes, Xtrain, Xtrain_raw, Ytrain_raw, Ztrain, Xvalid, Xvalid_raw, Yvalid_raw, Zvalid, labelEncoder, batch_size=20, epochs=150):

    run_name = '/massive_set_shuffle_split_aanvraag_1:1.2_heavy_aug_model_g_dropout_.6_lr.0003_nog_keer'
    LOG_DIR = f'./logs{run_name}'
    shutil.rmtree(LOG_DIR, ignore_errors=True)

    datagen = ImageDataGenerator(
        #     zoom_range=0.1,        # randomly zoom into images
        #     rotation_range=10,      # randomly rotate images in the range (degrees, 0 to 180)
        #     width_shift_range=0.1, # randomly shift images horizontally (fraction of total width)
        #     height_shift_range=0.1,# randomly shift images vertically (fraction of total height)
        #     shear_range=2.0,  # in degrees
        #     channel_shift_range=0.1,
        #     horizontal_flip=False,  # randomly flip images
        #     vertical_flip=False    # randomly flip images

        # Heavy augmentation
        zoom_range=0.15,        # randomly zoom into images
        rotation_range=15,      # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.15, # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.15,# randomly shift images vertically (fraction of total height)
        shear_range=4.0,  # in degrees
        channel_shift_range=0.15,
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False    # randomly flip images
    )

    tbCallBack = keras.callbacks.TensorBoard(
        log_dir=LOG_DIR,
        histogram_freq=0,
        write_graph=True,
        write_images=True
    )

    terminateCB = tf.keras.callbacks.TerminateOnNaN()



    t0 = time.time()

    # Img data
    # model = own_models.create_cnn(img_dim, num_classes)
    # history = train(model, Xtrain[0], Ytrain, Xvalid[0], Yvalid, batch_size, epochs)
    history = train_gen(model, Xtrain, Ztrain, Xvalid, Zvalid, batch_size, epochs)

    show_train_curves(history)

    difference = time.time() - t0
    print(f'time: {round(difference, 2)} seconds == {round(difference/60.0, 2)} minutes')


    # In[ ]:


    train_score = model.evaluate(Xtrain, Ztrain, verbose=1)
    print('Train loss:', round(train_score[0], 3))
    print(f'Train accuracy: {round(train_score[1] * 100, 2)}%')

    valid_score = model.evaluate(Xvalid, Zvalid, verbose=1)
    print('Test loss:', round(valid_score[0], 3))
    valid_acc_str = f'{round(valid_score[1] * 100, 2)}%'
    print(f'Test accuracy: {valid_acc_str}')


    # In[ ]:


    print(f"types: {classes}")

    print("train predictions, truth")
    predictions_train =  model.predict(Xtrain, verbose=1)
    show_prediction_list(predictions_train, Ztrain)

    print("test predictions, truth")
    predictions_valid = model.predict(Xvalid, verbose=1)
    show_prediction_list(predictions_valid, Zvalid)

    print("train set:")
    show_prediction_images(
        Xtrain_raw,
        Ztrain,
        predictions_train,
        Ytrain_raw[:, 2],
        labelEncoder,
        10
    )


    # In[ ]:


    print("test set:")
    show_prediction_images(
        Xvalid_raw,
        Zvalid,
        predictions_valid,
        Yvalid_raw[:, 2],
        labelEncoder,
        50
    )

    return predictions_train, predictions_valid


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


def show_results(threshold, Yvalid_raw, Zvalid, predictions, labelEncoder):
    ids = Yvalid_raw[:, 2]
    total = predictions.shape[0]
    [certain, uncertain] = split_bool_arrays(predictions, threshold)
    certain_count = np.sum(certain)
    uncertain_count = np.sum(uncertain)

    # Show counts of split
    certain_percentage = certain_count/total*100
    uncertain_percentage = uncertain_count/total*100
    counts_df = pd.DataFrame([
        [certain_count, uncertain_count, total],
        [certain_percentage, uncertain_percentage, 100.0]
    ],
        columns=['certain', 'uncertain', 'total'],
        index=['absolute', 'relative'])

    # Show metrics of splits
    [
        Ytrue_buckets,
        Ypred_buckets,
        ids_buckets,
    ] = split_uncertain(predictions, threshold, [Zvalid, predictions, ids])


    certain_not_empty = Ytrue_buckets[0].shape[0] > 0
    if certain_not_empty:
        y_true = labelEncoder.inverse_transform(Ytrue_buckets[0])
        y_pred = labelEncoder.inverse_transform(Ypred_buckets[0])
        certain_accuracy = accuracy_score(y_true, y_pred)
        certain_recall = recall_score(y_true, y_pred, pos_label='aanvraag')
        print(f'certain examples:\t\t{color.BOLD}{round(certain_percentage, 1)}%{color.END}', end='')
        print(f'\t accuracy: {color.BOLD}{round(certain_accuracy*100, 2)}%{color.END}', end='')
        print(f', aanvraag recall: {color.BOLD}{round(certain_recall*100, 2)}%{color.END}')

    else:
        print(f'certain examples:\t\t{color.BOLD}{round(certain_percentage, 1)}%{color.END}')
    print(f'uncertain examples:\t\t{color.BOLD}{round(uncertain_percentage, 1)}%{color.END}')

    print()
    print()
    print(counts_df.round(2))

    print()
    print()
    print(f'{color.BOLD}## Certain examples{color.END}')
    if certain_not_empty:
        certain_is_correct = show_reports(Ytrue_buckets[0], Ypred_buckets[0])
    else:
        print('no data')

    print()
    print()
    print(f'{color.BOLD}## Uncertain examples{color.END}')
    if Ytrue_buckets[1].shape[0] == 0:
        print('no data')
    else:
        uncertain_is_correct = show_reports(Ytrue_buckets[1], Ypred_buckets[1])

    print()
    print()
    print(f'{color.BOLD}## Certain errors{color.END}')
    certain_ids = ids_buckets[0]
    certain_ids.shape = (certain_ids.size, 1)
    certain_is_incorrect = np.invert(certain_is_correct)

    show_max = 50
    incorrect = certain_ids[certain_is_incorrect]
    print(f'incorrect ids[:{show_max}]:')
    print('\n'.join(incorrect[:show_max]))


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
