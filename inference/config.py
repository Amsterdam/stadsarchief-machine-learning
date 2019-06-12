import os

IIIF_ROOT = os.getenv("IIIF_ROOT")
assert IIIF_ROOT is not None

IIIF_IMAGE_DIR = os.getenv("IIIF_IMAGE_DIR")
assert IIIF_IMAGE_DIR is not None
os.makedirs(IIIF_IMAGE_DIR, exist_ok=True)

OUTPUT_DIR = os.getenv("OUTPUT_DIR")
assert OUTPUT_DIR is not None
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_DIR = os.getenv("MODEL_DIR")
assert MODEL_DIR is not None

MODEL_JSON = os.path.join(MODEL_DIR, './model.json')
MODEL_WEIGHTS = os.path.join(MODEL_DIR, './weights.h5')

TRANSFORM_DIR = os.path.join(MODEL_DIR, './transform/')
