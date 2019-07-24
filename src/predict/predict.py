import logging
from urllib.error import HTTPError

import numpy as np
from PIL import Image
from keras.engine.saving import model_from_json

from predict.config import TRANSFORM_DIR, IIIF_API_ROOT, IIIF_CACHE_DIR, MODEL_JSON, MODEL_WEIGHTS
from predict.iiif import IIIFClient
from processing.ImageFeatureEncoder import ImageFeatureEncoder
from processing.TargetEncoder import TargetEncoder

log = logging.getLogger(__name__)

iiifClient = IIIFClient(IIIF_API_ROOT, IIIF_CACHE_DIR)

imageEncoder = ImageFeatureEncoder()
imageEncoder.load(TRANSFORM_DIR)

targetEncoder = TargetEncoder()
targetEncoder.load(TRANSFORM_DIR)

print(f'Loading model: {MODEL_JSON}')
json_file = open(MODEL_JSON, 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

model.load_weights(MODEL_WEIGHTS)
print("Loaded model from disk")


def predict_single(path):
    image_data = np.array(Image.open(path))
    images_data = np.expand_dims(image_data, axis=0)  # set of examples of size 1

    # Transform data
    Xenc = imageEncoder.transform(images_data)

    result = model.predict(Xenc)  # [[prob_class1, prob_class2]]
    assert result.shape == (1, 2)

    category = targetEncoder.inverse_transform(result)[0][0]
    confidence = np.max(result)

    return [category, confidence]


