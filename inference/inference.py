from typing import List
import numpy as np
from PIL import Image
from keras.engine.saving import model_from_json

from config import IIIF_ROOT, IIIF_IMAGE_DIR, TRANSFORM_DIR, MODEL_JSON, MODEL_WEIGHTS
from iiif import IIIFClient
from src.processing.ImageFeatureEncoder import ImageFeatureEncoder
from src.processing.TargetEncoder import TargetEncoder

iiifClient = IIIFClient(IIIF_ROOT, IIIF_IMAGE_DIR)


imageEncoder = ImageFeatureEncoder()
imageEncoder.load(TRANSFORM_DIR)

targetEncoder = TargetEncoder()
targetEncoder.load(TRANSFORM_DIR)

print('Loading model')
json_file = open(MODEL_JSON, 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

model.load_weights(MODEL_WEIGHTS)
print("Loaded model from disk")


def run_inference_single(element):
    # Get data
    dim = [250, 250]
    stadsdeel_code = element.get('stadsdeel_code')
    dossier_nummer = element.get('dossier_nummer')
    document_id = element.get('document_id')
    path = iiifClient.get_image(stadsdeel_code, dossier_nummer, document_id, dim)

    image_data = np.array(Image.open(path))
    images_data = np.expand_dims(image_data, axis=0)  # set of examples of size 1

    # Transform data
    Xenc = imageEncoder.transform(images_data)

    result = model.predict(Xenc)  # [[prob_class1, prob_class2]]
    assert result.shape == (1, 2)

    category = targetEncoder.inverse_transform(result)[0][0]
    confidence = np.max(result)

    return [category, confidence]


