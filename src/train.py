import logging
import os
import shutil
import time
from dataclasses import dataclass

import keras
from keras import backend
import keras_metrics
from keras_preprocessing.image import ImageDataGenerator
import numpy as np
import tensorflow as tf

from src.data import DataGrouping, DataBlock

log = logging.getLogger(__name__)


@dataclass
class TrainingConfiguration:
    epochs: int
    batch_size: int
    optimizer: keras.optimizers.Optimizer
    data_generator: ImageDataGenerator
    class_weight: dict


def is_binary(model):
    n_classes = model.get_layer('output').output_shape[1]
    return n_classes == 1


def compile_model(model: keras.models.Model, optimzer: keras.optimizers.Optimizer):
    assert (backend.image_data_format() == 'channels_last')

    # if is_binary(model):
    #     loss= keras.losses.binary_crossentropy
    # else:
    loss = keras.losses.categorical_crossentropy

    model.compile(
        loss=loss,
        optimizer=optimzer,
        metrics=['accuracy', keras_metrics.binary_recall(label=0)]
    )
    return model


def train_model(
        model: keras.models.Model,
        data: DataGrouping,
        config: TrainingConfiguration,
        log_dir: str
):
    train_data: DataBlock = data.train
    valid_data: DataBlock = data.valid
    data_generator = config.data_generator

    run_name = 'massive_set_shuffle_split_aanvraag_1:1.2_heavy_aug_model_g_dropout_.6_lr.0003_nog_keer'
    run_log_dir = os.path.join(log_dir, run_name)
    shutil.rmtree(run_log_dir, ignore_errors=True)

    tensor_board_cb = keras.callbacks.TensorBoard(
        log_dir=run_log_dir,
        histogram_freq=0,
        write_graph=True,
        write_images=True
    )

    t0 = time.time()

    compile_model(model, config.optimizer)

    # actually fit
    print('Print Starting actual training')
    log.info('Log Starting actual training')
    history = model.fit_generator(
        data_generator.flow(
            train_data.images,
            train_data.labels,
            batch_size=config.batch_size
        ),
        steps_per_epoch=int(np.ceil(train_data.images.shape[0] / float(config.batch_size))),
        epochs=config.epochs,
        validation_data=(valid_data.images, valid_data.labels),
        class_weight=config.class_weight,
        workers=4,
        callbacks=[tensor_board_cb, tf.keras.callbacks.TerminateOnNaN()]
    )

    # show_train_curves(history)

    difference = time.time() - t0
    log.info(f'time: {round(difference, 2)} seconds == {round(difference/60.0, 2)} minutes')

    return history
