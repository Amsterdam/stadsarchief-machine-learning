"""
Flask app allowing annotations of yaml label files while serving image files
"""
import logging
import os
import sys

from src.annotation_server.server import create_app

log_level = logging.DEBUG
root = logging.getLogger()
root.setLevel(log_level)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(log_level)
root.addHandler(handler)

IIIF_API_ROOT = os.environ.get('IIIF_API_ROOT', 'https://images.data.amsterdam.nl/iiif/2/')

LABEL_DIR = os.environ.get('LABEL_DIR')
assert LABEL_DIR, 'path to label dir required (directory with <id>.yaml files)'

IMAGE_DIR = os.environ.get('IMAGE_DIR')  # path from Flask root_path (which is from this python file directory), not working directory!

PREDICTIONS_CSV = os.environ.get('PREDICTION_CSV')  # Optional

app = create_app(IIIF_API_ROOT, LABEL_DIR, IMAGE_DIR, PREDICTIONS_CSV)
