#!/usr/bin/env python
# coding: utf-8

"""
Aanvraag / besluit classifier training
"""
import keras
import numpy as np
import logging
import sys
from datasets.load_data import load_data_aanvraag, load_getting_started_data
from keras_preprocessing.image import ImageDataGenerator

from src.data import DataGrouping, DataBlock
from src.helper import feature_preprocess
from src.persistence.persistence import save_model_and_pre_processing
from src import helper
from src.stats import list_label_stats, show_prediction_list
from src.train import TrainingConfiguration, train_model
from src.util.np_size import display_MB
from src import models as own_models

gpus, devices = helper.tensorflow_gpu_supported()
print(f'gpus: {gpus}')
print(f'local devices: {devices}')

helper.gpu_memory_to_on_demand()

log = logging.getLogger(__name__)

log_level = logging.DEBUG
root = logging.getLogger()
root.setLevel(log_level)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(log_level)
root.addHandler(handler)


def load_data(img_dim: tuple) -> DataGrouping:
    [
        [Xtrain_raw, Ytrain_raw, Ztrain_raw],
        [Xvalid_raw, Yvalid_raw, Zvalid_raw],
        _test,
    ] = load_data_aanvraag(img_dim)

    log.info(f'shape Xtrain: {Xtrain_raw.shape}')
    log.info(f'shape Ytrain: {Ytrain_raw.shape}')
    log.info(f'shape Ztrain: {Ztrain_raw.shape}')

    log.info(f'shape Xvalid: {Xvalid_raw.shape}')
    log.info(f'shape Yvalid: {Yvalid_raw.shape}')
    log.info(f'shape Zvalid: {Zvalid_raw.shape}')

    log.warning(f'not using (hold out) test set of shape: {_test[0].shape}')

    log.info('training set size:')
    display_MB(Xtrain_raw)

    return DataGrouping(
        train=DataBlock(
            images_raw=Xtrain_raw,
            images=np.empty((0, img_dim[0], img_dim[1], 3)),
            labels_raw=Ztrain_raw,
            labels=np.empty((0, 2)),
            meta=Ytrain_raw,
        ),
        valid=DataBlock(
            images_raw=Xvalid_raw,
            images=np.empty((0, img_dim[0], img_dim[1], 3)),
            labels_raw=Zvalid_raw,
            labels=np.empty((0, 2)),
            meta=Yvalid_raw,
        ),
        test=None
    )


def pre_process(data: DataGrouping):
    train_data: DataBlock = data.train
    valid_data: DataBlock = data.valid

    Xtrain_raw = train_data.images_raw
    Xvalid_raw = valid_data.images_raw
    Ztrain_raw = train_data.labels_raw
    Zvalid_raw = valid_data.labels_raw

    # Feature
    Xtrain, Xvalid, imageEncoder = feature_preprocess(Xtrain_raw, Xvalid_raw)
    display_MB(Xtrain)

    # Labels
    classes, Ztrain, Zvalid, labelEncoder, labels = helper.label_preprocessing(Ztrain_raw, Zvalid_raw, 2)

    log.info('--- Train ---')
    list_label_stats(Ztrain_raw.ravel())

    log.info('--- Valid ---')
    list_label_stats(Zvalid_raw.ravel())

    log.info('--- transform ---')
    log.info(f'labels:\n{labels}')
    log.info(f'encoded:\n{labelEncoder.transform(labels).toarray()}')

    log.info('--- inverse transform ---')
    example = [[1, 0.5]]
    log.info(f'inverse transform for example {example}: {labelEncoder.inverse_transform(example)}')

    log.info('--- transform on subset of data ---')
    print_count = 10  # examples to show
    log.info(f'Ztrain_raw[:{print_count}]:\n{Ztrain_raw[:print_count]}')
    log.info(f'Ztrain:\n {Ztrain[:print_count]}')

    log.info('--- transformation shapes ---')
    log.info(f'Ztrain_raw shape:\t {Ztrain_raw.shape}')
    log.info(f'Ztrain shape:\t\t {Ztrain.shape}')
    log.info(f'Zvalid_raw shape:\t {Ztrain_raw.shape}')
    log.info(f'Zvalid shape:\t\t {Zvalid.shape}')

    train_data.images = Xtrain
    valid_data.images = Xvalid

    train_data.labels = Ztrain
    valid_data.labels = Zvalid

    return [data, {'imageEncoder': imageEncoder, 'labelEncoder': labelEncoder}]


def define_model(img_dim, num_classes=2):
    # model = own_models.create_cnn(img_dim, num_classes=num_classes)
    model = own_models.create_cnn_g(img_dim, num_classes=num_classes, drop_chance=0.6)
    return model


def evaluate_model(model, data: DataGrouping):
    train_score = model.evaluate(data.train.images, data.train.labels, verbose=1)
    log.info(f'Train loss: {round(train_score[0], 3)}')
    log.info(f'Train accuracy: {round(train_score[1] * 100, 2)}%')

    valid_score = model.evaluate(data.valid.images, data.valid.labels, verbose=1)
    log.info(f'Test loss: {round(valid_score[0], 3)}')
    valid_acc_str = f'{round(valid_score[1] * 100, 2)}%'
    log.info(f'Test accuracy: {valid_acc_str}')

    log.info("train predictions, truth")
    predictions_train = model.predict(data.train.images, verbose=1)
    show_prediction_list(predictions_train, data.train.labels)

    log.info("test predictions, truth")
    predictions_valid = model.predict(data.valid.images, verbose=1)
    show_prediction_list(predictions_valid, data.valid.labels)


def persist(model, encoders, model_out_dir):
    save_model_and_pre_processing(model, encoders.get('imageEncoder'), encoders.get('labelEncoder'), model_out_dir)


def main():
    model_out_dir = './output/model/'
    log_dir = './logs/'
    img_dim = (250, 250, 3)

    data_generator = ImageDataGenerator(
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False,    # randomly flip images

        # zoom_range=0.1,           # randomly zoom into images
        # rotation_range=10,        # randomly rotate images in the range (degrees, 0 to 180)
        # width_shift_range=0.1,    # randomly shift images horizontally (fraction of total width)
        # height_shift_range=0.1,   # randomly shift images vertically (fraction of total height)
        # shear_range=2.0,          # in degrees
        # channel_shift_range=0.1,

        # Heavy augmentation
        zoom_range=0.15,          # randomly zoom into images
        rotation_range=15,        # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.15,   # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.15,  # randomly shift images vertically (fraction of total height)
        shear_range=4.0,          # in degrees
        channel_shift_range=0.15,
    )

    learning_rate = 0.0003
    # learning_rate = 0.0001
    # learning_rate = 0.00003

    train_config = TrainingConfiguration(
        epochs=150,
        # epochs=4,
        batch_size=20,
        optimizer=keras.optimizers.Adam(lr=learning_rate),
        data_generator=data_generator,
        class_weight={
            0: 1.0,  # aanvraag
            1: 2.0,  # other error is x weight of aanvraag error (focus on learning recall)
        }
    )

    # Load
    data: DataGrouping = load_data(img_dim)

    # Pre process
    [data, encoders] = pre_process(data)

    # Model
    model = define_model(img_dim)
    model.summary()

    # Training
    train_model(model, data, train_config, log_dir)

    # Evaluate
    evaluate_model(model, data)

    # Persistence
    persist(model, encoders, model_out_dir)


if __name__ == '__main__':
    main()
