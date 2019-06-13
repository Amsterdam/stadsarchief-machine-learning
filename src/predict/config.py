import os

IIIF_API_ROOT = os.getenv("IIIF_API_ROOT")
assert IIIF_API_ROOT is not None

IIIF_CACHE_DIR = os.getenv("IIIF_CACHE_DIR")
assert IIIF_CACHE_DIR is not None
os.makedirs(IIIF_CACHE_DIR, exist_ok=True)

INPUT_JSON = os.getenv("INPUT_JSON")
assert INPUT_JSON is not None

OUTPUT_DIR = os.getenv("OUTPUT_DIR")
assert OUTPUT_DIR is not None
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_DIR = os.getenv("MODEL_DIR")
assert MODEL_DIR is not None

MODEL_JSON = os.path.join(MODEL_DIR, 'model.json')
MODEL_WEIGHTS = os.path.join(MODEL_DIR, 'weights.h5')

TRANSFORM_DIR = os.path.join(MODEL_DIR, 'transform/')
