"""
Flask app allowing annotations of yaml label files
"""
import logging
import os
import sys

from annotation_server.server import create_app

log_level = logging.DEBUG
root = logging.getLogger()
root.setLevel(log_level)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(log_level)
root.addHandler(handler)

iiif_api_root = os.environ.get('IIIF_API_ROOT')
assert iiif_api_root

label_dir = os.environ.get('LABEL_DIR')
assert label_dir, 'path to label dir required (directory with <id>.yaml files)'

image_dir = os.environ.get('IMAGE_DIR')  # path from Flask root_path (which is from this python file directory), not working directory!

app = create_app(iiif_api_root, label_dir, image_dir)
