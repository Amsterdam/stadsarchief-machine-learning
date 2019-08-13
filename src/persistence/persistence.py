from keras.engine.saving import model_from_json
import logging
import os

from src.processing.ImageFeatureEncoder import ImageFeatureEncoder
from src.processing.TargetEncoder import TargetEncoder

log = logging.getLogger(__name__)


def _get_paths(model_dir):
    model_json_path = os.path.join(model_dir, 'model.json')
    model_weights_path = os.path.join(model_dir, 'weights.h5')
    transform_dir = os.path.join(model_dir, 'transform/')
    return model_json_path, model_weights_path, transform_dir


def save_model_and_pre_processing(model, image_encoder, label_encoder, model_dir):
    os.makedirs(model_dir, exist_ok=True)
    model_json_path, model_weights_path, transform_dir = _get_paths(model_dir)
    os.makedirs(transform_dir, exist_ok=True)

    log.info(f'Writing image encoder to disk: {transform_dir}')
    image_encoder.save(transform_dir)

    log.info(f'Writing label encoder to disk: {transform_dir}')
    label_encoder.save(transform_dir)

    log.info(f'Writing model to {model_json_path} and {model_weights_path}')
    with open(model_json_path, "w") as json_file:
        json_file.write(model.to_json())
    model.save_weights(model_weights_path)


def load_model_and_pre_processing(model_dir):
    model_json_path, model_weights_path, transform_dir = _get_paths(model_dir)

    image_encoder = ImageFeatureEncoder()
    image_encoder.load(transform_dir)

    targetEncoder = TargetEncoder()
    targetEncoder.load(transform_dir)

    log.info(f'Loading model: {model_json_path}')
    json_file = open(model_json_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    model.load_weights(model_weights_path)
    log.info('Loaded model from disk')

    return [model, image_encoder, targetEncoder]
